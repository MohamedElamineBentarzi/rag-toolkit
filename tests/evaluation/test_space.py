"""SearchSpace: the tuner's input as declarative data. Hermetic."""
from __future__ import annotations

import pytest

from rag_blocks.core.errors import ConfigError
from rag_blocks.evaluation import SearchSpace, choice
from rag_blocks.evaluation.space import STAGE_KINDS


# -- choice() and its grid ----------------------------------------------


def test_a_choice_with_no_params_is_one_config():
    assert list(choice("markdown-aware").expand()) == [
        {"name": "markdown-aware", "params": {}}
    ]


def test_a_list_param_is_a_grid_axis():
    configs = list(choice("fixed", chunk_chars=[512, 1024]).expand())
    assert [c["params"]["chunk_chars"] for c in configs] == [512, 1024]


def test_a_scalar_param_is_one_value():
    configs = list(choice("index", representation="dense").expand())
    assert configs == [{"name": "index", "params": {"representation": "dense"}}]


def test_a_tuple_param_is_one_value_not_an_axis():
    # THE rule of this module. k_values=(1,5,10) configures one evaluator with
    # that tuple; a list would mean three separate trials. Params whose value
    # is genuinely a sequence are common, and expanding them would make those
    # untunable and silently wrong.
    configs = list(choice("ir", k_values=(1, 5, 10)).expand())
    assert len(configs) == 1
    assert configs[0]["params"]["k_values"] == (1, 5, 10)


def test_several_axes_take_the_cartesian_product():
    configs = list(
        choice("fixed", chunk_chars=[512, 1024], overlap_chars=[0, 64]).expand()
    )
    assert len(configs) == 4
    assert {(c["params"]["chunk_chars"], c["params"]["overlap_chars"]) for c in configs} == {
        (512, 0), (512, 64), (1024, 0), (1024, 64),
    }


def test_axes_and_fixed_params_mix():
    configs = list(choice("fixed", chunk_chars=[512, 1024], overlap_chars=0).expand())
    assert len(configs) == 2
    assert all(c["params"]["overlap_chars"] == 0 for c in configs)


def test_expansion_does_not_depend_on_keyword_order():
    # Two spellings of the same space must enumerate identically, or trial
    # ids would differ for identical pipelines.
    one = list(choice("fixed", a=[1, 2], b=[3, 4]).expand())
    two = list(choice("fixed", b=[3, 4], a=[1, 2]).expand())
    assert one == two


def test_an_empty_axis_fails_where_the_mistake_is():
    # product() with an empty sequence is empty, so this would silently
    # annihilate the whole grid — zero trials, no explanation.
    with pytest.raises(ConfigError, match="no values to try"):
        choice("fixed", chunk_chars=[])


def test_a_choice_needs_a_name():
    with pytest.raises(ConfigError):
        choice("")


# -- the space -----------------------------------------------------------


def test_a_single_stage_space_expands_to_its_choices():
    space = SearchSpace(chunker=[choice("fixed", chunk_chars=[512, 1024])])
    assert len(list(space.expand())) == 2 == len(space)


def test_stages_multiply():
    space = SearchSpace(
        chunker=[choice("fixed"), choice("markdown-aware")],
        generator=[choice("extractive")],
        refine=[[], [choice("keyword")]],
    )
    assert len(space) == 2 * 1 * 2 == len(list(space.expand()))


# -- nested choice: an encoder tuned inside a representation --------------


def test_a_nested_choice_becomes_a_sub_spec():
    # choice("dense", embedder=choice("hashing")) → the embedder param carries a
    # full {name, params} sub-spec, exactly what the builder resolves.
    configs = list(choice("dense", embedder=choice("hashing")).expand())
    assert configs == [
        {"name": "dense", "params": {"embedder": {"name": "hashing", "params": {}}}}
    ]


def test_a_nested_choice_grid_expands_the_outer_choice():
    # Tuning the encoder inside the representation: two dims → two dense specs.
    configs = list(
        choice("dense", embedder=choice("hashing", dimensions=[32, 64])).expand()
    )
    assert [c["params"]["embedder"]["params"]["dimensions"] for c in configs] == [32, 64]
    assert all(c["name"] == "dense" for c in configs)


def test_representations_is_a_chain_stage_over_nested_choices():
    space = SearchSpace(
        representations=[
            [choice("dense", embedder=choice("hashing", dimensions=[32, 64]))],
            [choice("dense", embedder=choice("hashing")), choice("lexical")],
        ],
    )
    # First option grids into 2 chains (32, 64); second is one chain → 3 total.
    combos = list(space.expand())
    assert len(combos) == 3
    assert all("representations" in c for c in combos)


def test_each_combination_names_every_stage():
    space = SearchSpace(chunker=[choice("fixed")], generator=[choice("extractive")])
    for combo in space.expand():
        assert set(combo) == {"chunker", "generator"}
        assert combo["chunker"]["name"] == "fixed"


def test_an_empty_space_yields_one_empty_combination():
    # "Vary nothing" is one pipeline: the defaults.
    assert list(SearchSpace().expand()) == [{}]
    assert len(SearchSpace()) == 1


# -- ordering: the reason the tuner needs no cache -----------------------


def test_dimensions_are_in_pipeline_order_not_alphabetical():
    # Load-bearing: alphabetically "parser" sorts after "generator", which
    # would make the MOST expensive stage vary fastest and re-parse the corpus
    # on nearly every trial.
    space = SearchSpace(
        generator=[choice("extractive")],
        parser=[choice("plaintext")],
        chunker=[choice("fixed")],
    )
    assert space.dimensions() == ["parser", "chunker", "generator"]


def test_expansion_keeps_shared_expensive_prefixes_adjacent():
    # The whole basis of "no new cache subsystem": trials sharing a parse run
    # back to back and inherit a warm blob cache.
    space = SearchSpace(
        chunker=[choice("fixed", chunk_chars=[512, 1024])],
        generator=[choice("extractive"), choice("extractive", max_context_chars=100)],
    )
    chunkers = [c["chunker"]["params"]["chunk_chars"] for c in space.expand()]
    # Grouped, not interleaved: [512, 512, 1024, 1024] rather than alternating.
    assert chunkers == [512, 512, 1024, 1024]


def test_expansion_is_deterministic():
    def build():
        return SearchSpace(
            chunker=[choice("fixed", chunk_chars=[512, 1024])],
            generator=[choice("extractive")],
        )

    assert list(build().expand()) == list(build().expand())


# -- chain stages --------------------------------------------------------


def test_a_chain_stage_takes_chains_including_the_empty_one():
    # "No reranker" is a real candidate — it is what a cross-encoder must beat
    # to earn its 180ms.
    space = SearchSpace(refine=[[], [choice("keyword")]])
    combos = list(space.expand())
    assert len(combos) == 2
    assert combos[0]["refine"] == []
    assert combos[1]["refine"] == [{"name": "keyword", "params": {}}]


def test_each_link_in_a_chain_expands_its_own_grid():
    # A chain of two choices with 2 params each is 4 chains, not 2.
    space = SearchSpace(
        refine=[[
            choice("neighbor-expander", window=[1, 2]),
            choice("score-threshold", min_score=[0.1, 0.2]),
        ]]
    )
    assert len(space) == 4
    for combo in space.expand():
        assert len(combo["refine"]) == 2


def test_a_multi_link_chain_keeps_its_order():
    space = SearchSpace(refine=[[choice("keyword"), choice("score-threshold")]])
    chain = next(space.expand())["refine"]
    assert [link["name"] for link in chain] == ["keyword", "score-threshold"]


def test_a_bare_choice_in_a_chain_stage_is_a_clear_error():
    # The likely typo: refine=[choice(...)] instead of refine=[[choice(...)]].
    with pytest.raises(ConfigError, match="takes chains"):
        SearchSpace(refine=[choice("keyword")])


def test_a_chain_holding_a_non_choice_fails_fast():
    with pytest.raises(ConfigError, match="expected choice"):
        SearchSpace(refine=[["keyword"]])


# -- fail fast -----------------------------------------------------------


def test_an_unknown_stage_fails_at_construction_and_lists_the_known_ones():
    with pytest.raises(ConfigError, match="unknown stage"):
        SearchSpace(chunkerr=[choice("fixed")])


def test_a_non_list_stage_fails_fast():
    with pytest.raises(ConfigError, match="must be a list"):
        SearchSpace(chunker=choice("fixed"))


def test_an_empty_stage_fails_fast():
    with pytest.raises(ConfigError, match="nothing to try"):
        SearchSpace(chunker=[])


def test_a_non_choice_option_fails_fast():
    with pytest.raises(ConfigError, match="takes choice"):
        SearchSpace(chunker=["fixed"])


def test_every_known_stage_maps_to_a_registry_kind():
    # If a stage key has no kind, the builder can't create it.
    assert all(isinstance(kind, str) and kind for kind in STAGE_KINDS.values())


def test_repr_says_how_many_trials_you_just_asked_for():
    space = SearchSpace(chunker=[choice("fixed", chunk_chars=[1, 2, 3])])
    assert "3 combinations" in repr(space)


def test_repr_is_ascii_so_printing_it_cannot_crash_on_windows():
    # Regression: a "→" here raised UnicodeEncodeError the moment anyone ran
    # print(repr(space)) on a cp1252 console — stdout has no backslashreplace
    # to save it, unlike stderr. A repr must never take down the program
    # printing it.
    repr(SearchSpace(chunker=[choice("fixed")])).encode("ascii")
