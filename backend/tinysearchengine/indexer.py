import json
import os
from dataclasses import dataclass, asdict, field
from enum import IntEnum
from io import UnsupportedOperation
from logging import getLogger
from mmap import mmap, PROT_READ, PROT_WRITE
from typing import TypeVar, Generic, Callable, List, Optional

import mmh3
from zstandard import ZstdDecompressor, ZstdCompressor, ZstdError

VERSION = 1
METADATA_CONSTANT = b"goodcoindex-tiny-search"
METADATA_SIZE = 4096

PAGE_SIZE = 4096


logger = getLogger(__name__)


class DocumentState(IntEnum):
    """
    The state of a document in the index.
    A value of None indicates an organic search result.
    """
    SYNCED_WITH_MAIN_INDEX = -2
    DELETED = -1
    FROM_USER = 2
    FROM_GOOGLE = 3
    FROM_WIKI = 4
    ORGANIC_APPROVED = 7
    FROM_USER_APPROVED = 8
    FROM_GOOGLE_APPROVED = 9
    FROM_WIKI_APPROVED = 10


CURATED_STATES = {state for state in DocumentState if state >= 7}


@dataclass
class Document:
    title: str
    url: str
    extract: str
    # TODO: Implement metascore
    score: Optional[float] = None
    term: Optional[str] = None
    state: Optional[int] = None

    def __init__(
        self,
        title: str,
        url: str,
        extract: str,
        score: Optional[float] = None,
        term: Optional[str] = None,
        state: Optional[int] = None,
    ):
        self.title = title if title is not None else ""
        self.url = url
        self.extract = extract if extract is not None else ""
        self.score = score
        self.term = term
        self.state = None if state is None else DocumentState(state)

    def as_tuple(self):
        """
        Covert a type to a tuple.
        Values at the end that are None can be truncated.
        """
        values = list(self.__dict__.values())
        if values[-1] is not None:
            values = values[-1].value

        while values[-1] is None:
            values = values[:-1]
        return tuple(values)


@dataclass
class TokenizedDocument(Document):
    tokens: List[str] = field(default_factory=list)



T = TypeVar('T')


class PageError(Exception):
    pass


@dataclass
class TinyIndexMetadata:
    version: int
    page_size: int
    num_pages: int
    item_factory: str

    def to_bytes(self) -> bytes:
        """
        Serialize the metadata to bytes.
        The metadata is a JSON object that is encoded to bytes.
        The metadata is padded to the page size.
        """
        metadata_bytes = METADATA_CONSTANT + \
            json.dumps(asdict(self)).encode('utf-8')
        assert len(metadata_bytes) <= METADATA_SIZE
        return metadata_bytes

    @staticmethod
    def from_bytes(data: bytes):
        """
        Deserialize the metadata from bytes.
        The metadata is a JSON object that is encoded to bytes.
        The metadata is padded to the page size.
        """
        constant_length = len(METADATA_CONSTANT)
        metadata_constant = data[:constant_length]
        if metadata_constant != METADATA_CONSTANT:
            raise ValueError("This doesn't seem to be an index file.")

        values = json.loads(data[constant_length:].decode('utf-8'))
        return TinyIndexMetadata(**values)

# Find the optimal number of items to fit in a page


def _binary_search_fitting_size(
        compressor: ZstdCompressor,
        page_size: int,
        items: list[T],
        lo: int,
        hi: int):
    """
    Binary search for the smallest number of items that can fit in a page.
    The items are serialized to JSON, compressed and then padded to the page size.
    The page size is the maximum size of the page.
    The items are a list of tuples.
    The lo and hi are the indices of the first and last item in the list.
    The compressor is the compressor to use to compress the items.
    The page_size is the maximum size of the page.
    """
    if lo > hi:
        return -1, None
    mid = (lo + hi) // 2
    compressed_data = compressor.compress(
        json.dumps(items[:mid]).encode('utf-8'))
    size = len(compressed_data)
    if size > page_size:
        return _binary_search_fitting_size(compressor, page_size, items, lo, mid - 1)
    else:
        potential_target, potential_data = _binary_search_fitting_size(
            compressor, page_size, items, mid + 1, hi)
        if potential_target != -1:
            return potential_target, potential_data
        else:
            return mid, compressed_data


def _trim_items_to_page(compressor: ZstdCompressor, page_size: int, items: list[T]):
    # Find max number of items that can fit in the page
    return _binary_search_fitting_size(compressor, page_size, items, 0, len(items))


def _get_page_data(page_size: int, items: list[T]):
    """
    Get the data for a page.
    The data is serialized to JSON, compressed and then padded to the page size.
    The page size is the maximum size of the page.
    The items are a list of tuples.
    The compressor is the compressor to use to compress the items.
    The page_size is the maximum size of the page.
    """
    compressor = ZstdCompressor()
    num_fitting, serialized_data = _trim_items_to_page(
        compressor, page_size, items)

    compressed_data = compressor.compress(
        json.dumps(items[:num_fitting]).encode('utf-8'))
    assert len(compressed_data) <= page_size, "The data shouldn't get bigger"
    return _pad_to_page_size(compressed_data, page_size)


def _pad_to_page_size(data: bytes, page_size: int):
    """
    Pad the data to the page size.
    The data is a bytes object.
    The page size is the maximum size of the page.
    """
    page_length = len(data)
    if page_length > page_size:
        raise PageError(
            f"Data is too big ({page_length}) for page size ({page_size})")
    padding = b'\x00' * (page_size - page_length)
    page_data = data + padding
    return page_data


class TinyIndex(Generic[T]):
    def __init__(self, item_factory: Callable[..., T], index_path, mode='r'):
        if mode not in { 'r', 'w' }:
            raise ValueError(f"Mode should be one of 'r' or 'w', got {mode}")
        
        with open(index_path, 'rb') as index_file:
            metadata_page = index_file.read(METADATA_SIZE)
        
        metadata_bytes = metadata_page.rstrip(b'\x00')
        metadata = TinyIndexMetadata.from_bytes(metadata_bytes)
        if metadata.item_factory != item_factory.__name__:
            raise ValueError(f"Metadata item factory '{metadata.item_factory}' in the index "
                             f"does not match the passed item factory: '{item_factory.__name__}'")
        
        self.item_factory = item_factory
        self.index_path = index_path
        self.mode = mode

        self.num_pages = metadata.num_pages
        self.page_size = metadata.page_size
        logger.info(f"Loaded index with {self.num_pages} pages and {self.page_size} page size")
        self.index_file = None
        self.mmap = None

    def __enter__(self):
        self.index_file = open(self.index_path, 'r+b')
        prot = PROT_READ if self.mode == 'r' else PROT_READ | PROT_WRITE
        self.mmap = mmap(self.index_file.fileno(), 0, prot=prot)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mmap.close()
        self.index_file.close()

    def retrieve(self, key: str) -> List[T]:
        """
        Retrieve all items that match the key.
        The key is a string that is hashed to get the page index.
        The page is then decompressed and deserialized using JSON.
        The items are then returned as a list.
        """
        index = self.get_key_page_index(key)
        logger.debug(f"Retrieving index {index}")
        page = self.get_page(index)
        return [item for item in page if item.term is None or item.term == key]

    def get_key_page_index(self, key) -> int:
        """
        Get the page index for a key.
        The key is a string that is hashed to get the page index.
        The page index is then returned.
        """
        key_hash = mmh3.hash(key, signed=False)
        return key_hash % self.num_pages
    
    def get_page(self, i) -> list[T]:
        """
        Get the page at index i, decompressing and deserializing it using JSON.
        The page is then returned as a list of tuples.
        """
        results = self.get_page_tuples(i)
        return [self.item_factory(*item) for item in results]
    
    def _get_page_tuples(self, i):
        """
        Get the page at index i, decompressing and deserializing it using JSON.
        The page is then returned as a list of tuples.
        """
        page_data = self.mmap[i * self.page_size + METADATA_SIZE: (i + 1) * self.page_size + METADATA_SIZE]
        decompressor = ZstdDecompressor()
        try:
            decompressed_data = decompressor.decompress(page_data)
        except ZstdError as e:
            raise PageError(f"Error decompressing page {i}: {e}")
        return json.loads(decompressed_data.decode('utf-8'))
    
    def store_in_page(self, page_index: int, values: list[T]):  
        """
        Store the values in the page at index page_index.
        The values are serialized to JSON, compressed and then padded to the page size.
        The page size is the maximum size of the page.
        The values are a list of tuples.
        """
        values_tuples = [value.as_tuple() for value in values]
        self._write_page(values_tuples, page_index)

    def _write_page(self, data, i: int):
        """
        Serialize the data using JSON, compress it and sote it at index i.
        If the data is too big, it will store the first items in the list 
        and discard the rest.
        """
        if self.mode != 'w':
            raise UnsupportedOperation("The file is open in read mode, and you can't write to it.")
        
        page_data = _get_page_data(self.page_size, data)
        logger.debug(f"Got page data of length {len(page_data)}")
        self.mmap[i * self.page_size + METADATA_SIZE:(i + 1) * self.page_size + METADATA_SIZE] = page_data
    
    @staticmethod
    def create(item_factory: Callable[..., T], index_path: str, num_pages: int, page_size: int):
        """
        Create a new index file.
        The index file is a file that contains the index data.
        The index file is created with the given number of pages and page size.
        The index file is created with the given item factory.
        """
        if os.path.isfile(index_path):
            raise FileExistsError(f"The index file {index_path} already exists.")
        
        metadata = TinyIndexMetadata(VERSION, page_size, num_pages, item_factory.__name__)
        metadata_bytes = metadata.to_bytes()
        metadata_padded = _pad_to_page_size(metadata_bytes, METADATA_SIZE)

        page_bytes = _get_page_data(page_size, [])

        with open(index_path, 'wb') as index_file:
            index_file.write(metadata_padded)
            for i in range(num_pages):
                index_file.write(page_bytes)
        
        return TinyIndex(item_factory, index_path, 'w')
        