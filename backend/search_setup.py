import os
import pickle
from pathlib import Path

from django.conf import settings
from django.core.cache import cache
from redis import Redis

from backend.tinysearchengine.completer import Completer
from backend.tinysearchengine.indexer import TinyIndex, Document
from backend.tinysearchengine.rank import HeuristicAndWikiRanker

completer = Completer()
index_path = Path(settings.DATA_PATH) / settings.INDEX_NAME
tiny_index = TinyIndex(item_factory=Document, index_path=index_path)
tiny_index.__enter__()


ranker = HeuristicAndWikiRanker(tiny_index, completer)
