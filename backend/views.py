from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from logging import getLogger
from typing import Optional
from urllib.parse import urlencode

from backend import justext
# import objgraph
# import psutil
# import requests
from django.conf import settings
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.forms import ModelForm, RadioSelect, CharField
from django.http import HttpResponseBadRequest
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.views.generic import DetailView, ListView
# from backend.justext.core import html_to_dom, ParagraphMaker, classify_paragraphs, revise_paragraph_classification, \
#     LENGTH_LOW_DEFAULT, STOPWORDS_LOW_DEFAULT, MAX_LINK_DENSITY_DEFAULT, NO_HEADINGS_DEFAULT, LENGTH_HIGH_DEFAULT, \
#     STOPWORDS_HIGH_DEFAULT, MAX_HEADING_DISTANCE_DEFAULT, DEFAULT_ENCODING, DEFAULT_ENC_ERRORS, preprocessor
# from requests.exceptions import RequestException

# from backend.crawler.app import stats_manager
# from backend.justext.utils import get_stoplist
# from backend.models import Curation, FlagCuration, DomainSubmission
from backend.search_setup import ranker, index_path
from backend.settings import NUM_EXTRACT_CHARS
from backend.tinysearchengine.indexer import Document, DocumentState, TinyIndex
from backend.tinysearchengine.rank import fix_document_state
from backend.tokenizer import tokenize
from backend.utils import add_term_infos, parse_url, validate_domain, float_or_none

MAX_CURATED_SCORE = 1_111_111.0

def _prepare_results(results: Optional[list[Document]]) -> Optional[dict[str, list[Document]]]:
    if results is None:
        return None

    grouped_domain_results = defaultdict(list)
    for result in results:
        domain = parse_url(result.url).netloc
        grouped_domain_results[domain].append(result)

    return dict(grouped_domain_results)

def index(request):
    activity, query, results = _get_results_and_activity(request)
    return render(request, 'index.html', {
        "results": _prepare_results(results),
        "query": query,
        "user": request.user,
        "activity": activity,
        "footer_links": settings.FOOTER_LINKS,
    })

def _get_results_and_activity(request):
    query = request.GET.get("q")
    if query:
        # There may be extra results in the request that we need to add in
        # format is ?enhanced=google&title=title1&url=url1&extract=extract1&title=title2&url=url2&extract=extract2
        # source = request.GET.get("enhanced", "unknown")
        titles = request.GET.getlist(f"title")
        urls = request.GET.getlist(f"url")
        extracts = request.GET.getlist(f"extract")

        term = " ".join(tokenize(query))

        # For now, we only support the Google source
        additional_results = [
            Document(title=title, url=url, extract=extract, score=100.0 * 2 ** -i, term=term, state=DocumentState.FROM_GOOGLE)
            for i, (title, url, extract) in enumerate(zip(titles, urls, extracts))
        ]

        results = ranker.search(query, additional_results=additional_results)
        activity = None
    else:
        results = None
        activity = Curation.objects.filter(flag_curation_set__isnull=True).order_by("-timestamp")[:8]
    return activity, query, results