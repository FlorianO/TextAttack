import flair
from flair.data import Sentence
from flair.models import SequenceTagger
import lru
import nltk

import textattack
from textattack.constraints import Constraint
from textattack.shared.validators import transformation_consists_of_word_swaps

# Set global flair device to be TextAttack's current device
flair.device = textattack.shared.utils.device


def load_flair_upos_fast():
    """Loads flair 'upos-fast' SequenceTagger.

    This is a temporary workaround for flair v0.6. Will be fixed when
    flair pushes the bug fix.
    """
    import pathlib
    import warnings

    from flair import file_utils
    import torch

    hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"
    upos_path = "/".join([hu_path, "upos-fast", "en-upos-ontonotes-fast-v0.4.pt"])
    model_path = file_utils.cached_path(upos_path, cache_dir=pathlib.Path("models"))
    model_file = SequenceTagger._fetch_model(model_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # load_big_file is a workaround by https://github.com/highway11git to load models on some Mac/Windows setups
        # see https://github.com/zalandoresearch/flair/issues/351
        f = file_utils.load_big_file(str(model_file))
        state = torch.load(f, map_location="cpu")
    model = SequenceTagger._init_model_with_state_dict(state)
    model.eval()
    model.to(textattack.shared.utils.device)
    return model


class PartOfSpeech(Constraint):
    """Constraints word swaps to only swap words with the same part of speech.
    Uses the NLTK universal part-of-speech tagger by default. An implementation
    of `<https://arxiv.org/abs/1907.11932>`_ adapted from
    `<https://github.com/jind11/TextFooler>`_.

    POS tagger from Flair `<https://github.com/flairNLP/flair>` also available

    Args:
        tagger_type (str): Name of the tagger to use (available choices: "nltk", "flair").
        tagset (str): tagset to use for POS tagging
        allow_verb_noun_swap (bool): If `True`, allow verbs to be swapped with nouns and vice versa.
        compare_against_original (bool): If `True`, compare against the original text.
            Otherwise, compare against the most recent text.
    """

    def __init__(
        self,
        tagger_type="nltk",
        tagset="universal",
        allow_verb_noun_swap=True,
        compare_against_original=True,
    ):
        super().__init__(compare_against_original)
        self.tagger_type = tagger_type
        self.tagset = tagset
        self.allow_verb_noun_swap = allow_verb_noun_swap

        self._pos_tag_cache = lru.LRU(2 ** 14)
        if tagger_type == "flair":
            if tagset == "universal":
                self._flair_pos_tagger = load_flair_upos_fast()
            else:
                self._flair_pos_tagger = SequenceTagger.load("pos-fast")

    def clear_cache(self):
        self._pos_tag_cache.clear()

    def _can_replace_pos(self, pos_a, pos_b):
        return (pos_a == pos_b) or (
            self.allow_verb_noun_swap and set([pos_a, pos_b]) <= set(["NOUN", "VERB"])
        )

    def _get_pos(self, before_ctx, word, after_ctx):
        context_words = before_ctx + [word] + after_ctx
        context_key = " ".join(context_words)
        if context_key in self._pos_tag_cache:
            word_list, pos_list = self._pos_tag_cache[context_key]
        else:
            if self.tagger_type == "nltk":
                word_list, pos_list = zip(
                    *nltk.pos_tag(context_words, tagset=self.tagset)
                )

            if self.tagger_type == "flair":
                context_key_sentence = Sentence(context_key)
                self._flair_pos_tagger.predict(context_key_sentence)
                word_list, pos_list = textattack.shared.utils.zip_flair_result(
                    context_key_sentence
                )

            self._pos_tag_cache[context_key] = (word_list, pos_list)

        # idx of `word` in `context_words`
        assert word in word_list, "POS list not matched with original word list."
        word_idx = word_list.index(word)
        return pos_list[word_idx]

    def _check_constraint(self, transformed_text, reference_text):
        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply part-of-speech constraint without `newly_modified_indices`"
            )

        for i in indices:
            reference_word = reference_text.words[i]
            transformed_word = transformed_text.words[i]
            before_ctx = reference_text.words[max(i - 4, 0) : i]
            after_ctx = reference_text.words[
                i + 1 : min(i + 4, len(reference_text.words))
            ]
            ref_pos = self._get_pos(before_ctx, reference_word, after_ctx)
            replace_pos = self._get_pos(before_ctx, transformed_word, after_ctx)
            if not self._can_replace_pos(ref_pos, replace_pos):
                return False

        return True

    def check_compatibility(self, transformation):
        return transformation_consists_of_word_swaps(transformation)

    def extra_repr_keys(self):
        return [
            "tagger_type",
            "tagset",
            "allow_verb_noun_swap",
        ] + super().extra_repr_keys()
