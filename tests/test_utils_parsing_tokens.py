"""Token repair must not glue ordinary words together.

The repair exists for one artifact: a word broken by a stray separator,
e.g. "Mi_iti" for "Misiti". The old rule fired on any word of three
letters or less, which mangled ordinary Italian and English phrases —
"per il mese" became "peril mese" and "via roma" became "viaroma".
"""
from __future__ import annotations

import pytest

from archiver.utils_filename import normalize_separators
from archiver.utils_parsing import split_and_repair_tokens


def test_broken_word_is_still_repaired():
    # The documented case: a capitalised fragment followed by a lowercase tail.
    assert split_and_repair_tokens("Mi_iti") == ["Miiti"]


@pytest.mark.parametrize("stem, expected", [
    ("per il mese di marzo", ["per", "il", "mese", "di", "marzo"]),
    ("for the office", ["for", "the", "office"]),
    ("con tre spirali nere su sfondo verde",
     ["con", "tre", "spirali", "nere", "su", "sfondo", "verde"]),
    ("via roma 12", ["via", "roma", "12"]),
    ("due case a san marco", ["due", "case", "a", "san", "marco"]),
])
def test_ordinary_short_words_are_left_alone(stem, expected):
    assert split_and_repair_tokens(stem) == expected


def test_a_capitalised_tail_is_not_a_broken_word():
    # "Via Roma" is two words, not one word split in half.
    assert split_and_repair_tokens("Via Roma 12") == ["Via", "Roma", "12"]


def test_a_capitalised_function_word_is_not_merged_either():
    # Erring towards leaving a name split: a wrong merge corrupts it for good,
    # a missed one still reads.
    assert split_and_repair_tokens("Il mese di marzo") == ["Il", "mese", "di", "marzo"]


def test_the_proposed_name_survives_separator_normalisation():
    # End to end through the function that builds the archived filename.
    name = "simbolo antico triskelion con tre spirali nere su sfondo verde.jpg"
    assert normalize_separators(name, sep="dash") == (
        "simbolo-antico-triskelion-con-tre-spirali-nere-su-sfondo-verde.jpg"
    )
