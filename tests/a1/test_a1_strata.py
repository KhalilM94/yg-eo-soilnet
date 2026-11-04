"""The FAO carbonate classes, cut exactly as the paper figures cut them.

A value on a boundary is the case that matters: the paper figures put 2 %, 10 % and 25 % in the class
above, and a breakdown that put them in the class below would quietly disagree with every figure.
"""

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.a1.strata import FAO_CARBONATE_EDGES, FAO_CARBONATE_LABELS, carbonate_class


def _classes(*values):
    return carbonate_class(pd.Series(values, dtype=float)).tolist()


@pytest.mark.parametrize("edge, above", [(2.0, "2-10"), (10.0, "10-25"), (25.0, ">=25")])
def test_a_value_exactly_on_a_boundary_goes_to_the_class_above(edge, above):
    assert _classes(edge) == [above]


@pytest.mark.parametrize("value, below", [(1.999, "<2"), (9.999, "2-10"), (24.999, "10-25")])
def test_a_value_just_under_a_boundary_stays_in_the_class_below(value, below):
    assert _classes(value) == [below]


def test_the_ends_of_the_range_fall_in_the_end_classes():
    assert _classes(0.0, 60.0, 100.0) == ["<2", ">=25", ">=25"]


def test_the_labels_are_the_fao_classes():
    assert FAO_CARBONATE_EDGES == (2.0, 10.0, 25.0)
    assert FAO_CARBONATE_LABELS == ("<2", "2-10", "10-25", ">=25")


def test_the_classes_are_ordered_from_least_to_most_calcareous():
    classes = carbonate_class(pd.Series([30.0, 1.0, 12.0, 5.0]))

    assert classes.cat.ordered
    assert list(classes.cat.categories) == list(FAO_CARBONATE_LABELS)
    assert classes.sort_values().tolist() == ["<2", "2-10", "10-25", ">=25"]
    assert classes.iloc[0] > classes.iloc[2]


def test_every_class_is_a_category_even_when_no_value_falls_in_it():
    assert list(carbonate_class(pd.Series([5.0])).cat.categories) == list(FAO_CARBONATE_LABELS)


def test_a_missing_value_has_no_class():
    classes = carbonate_class(pd.Series([np.nan, 3.0, None]))
    assert classes.isna().tolist() == [True, False, True]


def test_the_index_is_kept():
    values = pd.Series([1.0, 30.0], index=["p7", "p3"])
    assert carbonate_class(values).index.tolist() == ["p7", "p3"]


def test_other_classes_can_be_asked_for():
    classes = carbonate_class(pd.Series([4.0, 6.0]), edges=[5], labels=["low", "high"])
    assert classes.tolist() == ["low", "high"]


@pytest.mark.parametrize("edges", [[10, 2, 25], [2, 2, 25], []])
def test_boundaries_that_do_not_rise_are_refused(edges):
    with pytest.raises(ValueError, match="increasing"):
        carbonate_class(pd.Series([1.0]), edges=edges, labels=["x"] * (len(edges) + 1))


def test_a_class_without_a_name_is_refused():
    with pytest.raises(ValueError, match="one name per class"):
        carbonate_class(pd.Series([1.0]), edges=[2, 10, 25], labels=["a", "b", "c"])
