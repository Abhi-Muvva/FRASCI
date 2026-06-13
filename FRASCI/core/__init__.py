# frasci.core — shared infrastructure used by all coupling levels
from FRASCI.core.fragment import (
    fragment_by_sliding_window,
    extract_fragment_integrals,
    fragment_electron_count,
)
from FRASCI.core.trimci_adapter import (
    FragmentResult,
    solve_fragment_trimci,
    solve_fragment_exact,
)
from FRASCI.core.results import FragmentedRunResult
from FRASCI.core.analysis import (
    determinant_summary,
    iteration_summary,
    convergence_summary,
)
