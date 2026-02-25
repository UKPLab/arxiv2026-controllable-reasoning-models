# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of all """

from .constraint_checker import *


_KEYWORD = "keywords:"

_LANGUAGE = "language:"

_LENGTH = "length_constraint_checkers:"

_CONTENT = "detectable_content:"

_FORMAT = "detectable_format:"

_MULTITURN = "multi-turn:"

_COMBINATION = "combination:"

_STARTEND = "startend:"

_CHANGE_CASES = "change_case:"

_PUNCTUATION = "punctuation:"

INSTRUCTION_DICT = {
    _KEYWORD + "existence": KeywordChecker,
    _KEYWORD + "frequency": KeywordFrequencyChecker,
    # TODO(jeffreyzhou): make a proper set of sentences to choose from
    # _KEYWORD + "key_sentences": KeySentenceChecker,
    _KEYWORD + "forbidden_words": ForbiddenWords,
    # _KEYWORD + "letter_frequency": LetterFrequencyChecker,
    _LANGUAGE + "response_language": ResponseLanguageChecker,
    # _LENGTH + "number_sentences": NumberOfSentences,
    # _LENGTH + "number_paragraphs": ParagraphChecker,
    _LENGTH + "number_words": NumberOfWords,
    # _LENGTH + "nth_paragraph_first_word": ParagraphFirstWordCheck,
    # _CONTENT + "number_placeholders": PlaceholderChecker,
    # _CONTENT + "postscript": PostscriptChecker,
    _FORMAT + "number_bullet_lists": BulletListChecker,
    # TODO(jeffreyzhou): Pre-create paragraph or use prompt to replace
    # _CONTENT + "rephrase_paragraph": RephraseParagraph,
    # _FORMAT + "constrained_response": ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": (HighlightSectionChecker),
    _FORMAT + "multiple_sections": SectionChecker,
    # TODO(tianjianlu): Re-enable rephrasing with preprocessing the message.
    # _FORMAT + "rephrase": RephraseChecker,
    # _FORMAT + "json_format": JsonFormat,
    # _FORMAT + "title": TitleChecker,
    # TODO(tianjianlu): Re-enable with specific prompts.
    # _MULTITURN + "constrained_start": ConstrainedStartChecker,
    # _COMBINATION + "two_responses": TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": RepeatPromptThenAnswer,
    _STARTEND + "end_checker": EndChecker,
    _STARTEND + "quotation": QuotationChecker,
    _CHANGE_CASES + "capital_word_frequency": CapitalWordFrequencyChecker,
    _CHANGE_CASES + "english_capital": CapitalLettersEnglishChecker,
    _CHANGE_CASES + "english_lowercase": LowercaseLettersEnglishChecker,
    _PUNCTUATION + "no_comma": CommaChecker,
}

INSTRUCTION_CONFLICTS = {
    _KEYWORD + "existence": {_KEYWORD + "existence"},
    _KEYWORD + "frequency": {_KEYWORD + "frequency"},
    # TODO(jeffreyzhou): make a proper set of sentences to choose from
    # _KEYWORD + "key_sentences": KeySentenceChecker,
    _KEYWORD + "forbidden_words": {_KEYWORD + "forbidden_words"},
    #_KEYWORD + "letter_frequency": {_KEYWORD + "letter_frequency"},
    _LANGUAGE + "response_language": {
        _LANGUAGE + "response_language",
        _FORMAT + "multiple_sections",
        _KEYWORD + "existence",
        _KEYWORD + "frequency",
        _KEYWORD + "forbidden_words",
        _STARTEND + "end_checker",
        _CHANGE_CASES + "english_capital",
        _CHANGE_CASES + "english_lowercase",
    },
    # _LENGTH + "number_sentences": {_LENGTH + "number_sentences"},
    # _LENGTH + "number_paragraphs": {
    #     _LENGTH + "number_paragraphs",
    #     _LENGTH + "nth_paragraph_first_word",
    #     _LENGTH + "number_sentences",
    #     _LENGTH + "nth_paragraph_first_word",
    # },
    _LENGTH + "number_words": {_LENGTH + "number_words"},
    # _LENGTH + "nth_paragraph_first_word": {
    #     _LENGTH + "nth_paragraph_first_word",
    #     _LENGTH + "number_paragraphs",
    # },
    # _CONTENT + "number_placeholders": {_CONTENT + "number_placeholders"},
    # _CONTENT + "postscript": {_CONTENT + "postscript"},
    _FORMAT + "number_bullet_lists": {_FORMAT + "number_bullet_lists"},
    # TODO(jeffreyzhou): Pre-create paragraph or use prompt to replace
    # _CONTENT + "rephrase_paragraph": RephraseParagraph,
    # _FORMAT + "constrained_response": set(INSTRUCTION_DICT.keys()),
    _FORMAT + "number_highlighted_sections": {_FORMAT + "number_highlighted_sections"},
    _FORMAT + "multiple_sections": {
        _FORMAT + "multiple_sections",
        _LANGUAGE + "response_language",
        _FORMAT + "number_highlighted_sections",
    },
    # TODO(tianjianlu): Re-enable rephrasing with preprocessing the message.
    # _FORMAT + "rephrase": RephraseChecker,
    # _FORMAT + "json_format": set(INSTRUCTION_DICT.keys()).difference(
    #     {_KEYWORD + "forbidden_words", _KEYWORD + "existence"}
    # ),
    # _FORMAT + "title": {_FORMAT + "title"},
    # TODO(tianjianlu): Re-enable with specific prompts.
    # _MULTITURN + "constrained_start": ConstrainedStartChecker,
    # _COMBINATION + "two_responses": set(INSTRUCTION_DICT.keys()).difference(
    #     {
    #         _KEYWORD + "forbidden_words",
    #         _KEYWORD + "existence",
    #         _LANGUAGE + "response_language",
    #         _FORMAT + "title",
    #         _PUNCTUATION + "no_comma",
    #     }
    # ),
    _COMBINATION + "repeat_prompt": set(INSTRUCTION_DICT.keys()).difference(
        {
            _KEYWORD + "existence", 
            # _FORMAT + "title", 
            _PUNCTUATION + "no_comma"}
    ),
    _STARTEND + "end_checker": {_STARTEND + "end_checker"},
    _CHANGE_CASES + "capital_word_frequency": {
        _CHANGE_CASES + "capital_word_frequency",
        _CHANGE_CASES + "english_lowercase",
        _CHANGE_CASES + "english_capital",
    },
    _CHANGE_CASES + "english_capital": {_CHANGE_CASES + "english_capital"},
    _CHANGE_CASES + "english_lowercase": {
        _CHANGE_CASES + "english_lowercase",
        _CHANGE_CASES + "english_capital",
    },
    _PUNCTUATION + "no_comma": {_PUNCTUATION + "no_comma"},
    _STARTEND + "quotation": {
        _STARTEND + "quotation",
        # _FORMAT + "title"
        },
}


def conflict_make(conflicts):
    """Makes sure if A conflicts with B, B will conflict with A.

    Args:
      conflicts: Dictionary of potential conflicts where key is instruction id
        and value is set of instruction ids that it conflicts with.

    Returns:
      Revised version of the dictionary. All constraint_checker conflict with
      themselves. If A conflicts with B, B will conflict with A.
    """
    for key in conflicts:
        for k in conflicts[key]:
            conflicts[k].add(key)
        conflicts[key].add(key)
    return conflicts

DOUBLE_CONSTRAINT = [
    ('combination:repeat_prompt', 'keywords:forbidden_words'),
    ('change_case:english_lowercase', 'keywords:frequency'),
    ('change_case:english_lowercase', 'keywords:existence'),
    # ('detectable_format:multiple_sections', 'detectable_format:number_bullet_lists'),
    ('language:response_language', 'startend:quotation'),
    ('keywords:existence', 'punctuation:no_comma'),
    ('change_case:english_lowercase', 'detectable_format:number_highlighted_sections'),
    ('keywords:forbidden_words', 'startend:quotation'),
    # ('change_case:capital_word_frequency', 'change_case:capital_word_frequency'),
    ('change_case:english_capital', 'keywords:forbidden_words'),
    ('language:response_language', 'punctuation:no_comma'),
    ('change_case:capital_word_frequency', 'punctuation:no_comma'),
    ('keywords:forbidden_words', 'punctuation:no_comma'),
    ('keywords:forbidden_words', 'keywords:frequency'),
    ('detectable_format:number_highlighted_sections', 'startend:quotation'),
    ('change_case:english_lowercase', 'detectable_format:number_bullet_lists'),
    ('detectable_format:number_bullet_lists', 'keywords:forbidden_words'),
    ('detectable_format:multiple_sections', 'startend:quotation'),
    ('keywords:frequency', 'keywords:frequency'),
    ('combination:repeat_prompt', 'keywords:existence'),
    ('change_case:english_lowercase', 'keywords:forbidden_words'),
    ('keywords:frequency', 'punctuation:no_comma'),
    ('combination:repeat_prompt', 'keywords:frequency'),
    ('combination:repeat_prompt', 'punctuation:no_comma'),
    ('keywords:existence', 'keywords:forbidden_words'),
    ('detectable_format:number_highlighted_sections', 'keywords:frequency'),
    ('detectable_format:number_highlighted_sections', 'punctuation:no_comma'),
    ('detectable_format:number_highlighted_sections', 'keywords:existence'),
    ('keywords:frequency', 'startend:end_checker'),
    ('punctuation:no_comma', 'startend:end_checker'),
    ('detectable_format:number_bullet_lists', 'startend:quotation'),
    ('change_case:english_capital', 'punctuation:no_comma'),
    # ('change_case:english_capital', 'detectable_format:multiple_sections'),
    ('change_case:english_capital', 'keywords:existence'),
    ('detectable_format:multiple_sections', 'startend:end_checker'),
    ('keywords:existence', 'startend:quotation'),
    ('change_case:english_lowercase', 'startend:quotation'),
    ('change_case:english_capital', 'detectable_format:number_highlighted_sections'),
    ('change_case:english_capital', 'startend:end_checker'),
    ('detectable_format:number_bullet_lists', 'punctuation:no_comma')
]

TRIPLE_CONSTRAINT = [
    ('change_case:capital_word_frequency', 'length_constraint_checkers:number_words', 'keywords:frequency'),
    ('combination:repeat_prompt', 'keywords:existence', 'keywords:forbidden_words'),
    ('detectable_format:number_highlighted_sections', 'punctuation:no_comma', 'startend:end_checker'),
    ('detectable_format:number_highlighted_sections', 'keywords:existence', 'punctuation:no_comma'),
    ('change_case:capital_word_frequency', 'keywords:frequency', 'punctuation:no_comma'),
    ('keywords:existence', 'keywords:frequency', 'keywords:frequency'),
    ('detectable_format:number_highlighted_sections', 'keywords:existence', 'keywords:frequency'),
    ('detectable_format:number_bullet_lists', 'keywords:forbidden_words', 'keywords:frequency'),
    ('detectable_format:multiple_sections', 'keywords:existence', 'keywords:frequency'),
    ('detectable_format:number_bullet_lists', 'keywords:existence', 'length_constraint_checkers:number_words'),
    ('combination:repeat_prompt', 'length_constraint_checkers:number_words', 'keywords:forbidden_words'),
    ('change_case:english_lowercase', 'keywords:existence', 'keywords:frequency'),
    ('keywords:existence', 'length_constraint_checkers:number_words', 'startend:quotation'),
    ('combination:repeat_prompt', 'detectable_format:number_highlighted_sections', 'length_constraint_checkers:number_words'),
    ('change_case:english_capital', 'punctuation:no_comma', 'startend:quotation')
]