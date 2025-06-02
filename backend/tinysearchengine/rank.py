import html
import json
import math
import re
import urllib
from abc import abstractmethod
from collections import defaultdict
from datetime import timedelta
from logging import getLogger
from operator import itemgetter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from backend.format import get_query_regex
from backend.hn_top_domains_filtered import DOMAINS
from backend.tinysearchengine.completer import Completer
from backend.tinysearchengine.indexer import Document, DocumentState,TinyIndex
from backend.tokenizer import tokenize, get_bigrams
from backend.utils import request_cache


logger = getLogger(__name__)


MAX_QUERY_CHARS = 100
MATCH_SCORE_THRESHOLD = 0.0
SCORE_THRESHOLD = 0.0
LENGTH_PENALTY = 0.04
MATCH_EXPONENT = 2
DOMAIN_SCORE_SMOOTHING = 0.1
HTTPS_STRING = "https://"