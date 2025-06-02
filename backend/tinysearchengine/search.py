import dataclasses
from logging import getLogger

from ninja import NinjaAPI

from backend.format import format_result
from backend.tinysearchengine.indexer import Document
