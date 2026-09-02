"""Virtual arrays subpackage for data-conduit."""

from data_conduit.virtualarrays.base_queries import ulookup
from data_conduit.virtualarrays.custom_lookup_template import LookupAccessorConstructor
from data_conduit.virtualarrays.virtual_arrays_core import (
    construct_data_array,
    construct_lookup_array,
    update_data_array,
)

__all__ = [
    "LookupAccessorConstructor",
    "construct_data_array",
    "construct_lookup_array",
    "ulookup",
    "update_data_array",
]
