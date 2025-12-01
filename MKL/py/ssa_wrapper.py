"""
SSA Python Wrapper
==================

Ctypes wrapper for ssa_opt.h MKL-optimized SSA implementation.

Usage:
    from ssa_wrapper import SSA, MSSA
    
    # Univariate SSA
    ssa = SSA(signal, L=100)
    ssa.decompose(k=20)
    trend = ssa.reconstruct([0])
    forecast = ssa.forecast([0, 1, 2], n_forecast=50)
    
    # Multivariate SSA
    mssa = MSSA(X, L=100)  # X is (M, N) array
    mssa.decompose(k=20)
    series0_trend = mssa.reconstruct(series_idx=0, group=[0])

Requirements:
    - libssa.so (or libssa.dll on Windows) in same directory or LD_LIBRARY_PATH
    - numpy
"""

import ctypes
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Optional, Union, Tuple

print(f"Python: {sys.version}")
print(f"Platform: {sys.platform}")

# Check if ssa.dll exists
dll_path = Path(__file__).parent / "ssa.dll"
print(f"Looking for: {dll_path}")
print(f"Exists: {dll_path.exists()}")

# Check MKL paths
mkl_paths = [
    r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
    r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
]
for p in mkl_paths:
    print(f"MKL path {p}: exists={os.path.exists(p)}")
    if os.path.exists(p):
        os.add_dll_directory(p)

# Try loading directly
import ctypes
try:
    lib = ctypes.CDLL(str(dll_path))
    print("SUCCESS!")
except OSError as e:
    print(f"FAILED: {e}")

# ============================================================================
# Windows MKL DLL paths (must be set BEFORE loading the library)
# ============================================================================

if sys.platform == "win32":
    mkl_paths = [
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
    ]
    for p in mkl_paths:
        if os.path.exists(p):
            os.add_dll_directory(p)

# ============================================================================
# Load shared library
# ============================================================================

def _load_library():
    """Find and load the SSA shared library."""
    # Search paths
    search_paths = [
        Path(__file__).parent / "libssa.so",
        Path(__file__).parent / "libssa.dll",
        Path(__file__).parent / "ssa.dll",     
        Path(".") / "libssa.so",
        Path(".") / "libssa.dll",
        Path(".") / "ssa.dll",                  
        "libssa.so",
        "libssa.dll",
        "ssa.dll",                             
    ]
    
    for path in search_paths:
        try:
            return ctypes.CDLL(str(path))
        except OSError:
            continue
    
    raise RuntimeError(
        "Could not load libssa.so/.dll. Build it with:\n"
        "  gcc -shared -fPIC -O3 -o libssa.so ssa_wrapper.c \\\n"   
        "      -DSSA_OPT_IMPLEMENTATION -DSSA_USE_MKL \\\n"
        "      -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_rt -lm"
    )

_lib = _load_library()

# ============================================================================
# Opaque struct definitions (oversized buffers for safety)
# ============================================================================

class _SSA_Opt(ctypes.Structure):
    """Opaque wrapper for SSA_Opt C struct."""
    _fields_ = [("_opaque", ctypes.c_char * 1024)]

class _MSSA_Opt(ctypes.Structure):
    """Opaque wrapper for MSSA_Opt C struct."""
    _fields_ = [("_opaque", ctypes.c_char * 1024)]

class _SSA_LRF(ctypes.Structure):
    """Opaque wrapper for SSA_LRF C struct."""
    _fields_ = [("_opaque", ctypes.c_char * 256)]

class _SSA_ComponentStats(ctypes.Structure):
    """Opaque wrapper for SSA_ComponentStats C struct."""
    _fields_ = [("_opaque", ctypes.c_char * 512)]

# ============================================================================
# Function signatures
# ============================================================================

# Pointer types
c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)

# --- SSA functions ---
_lib.ssa_opt_init.argtypes = [ctypes.POINTER(_SSA_Opt), c_double_p, ctypes.c_int, ctypes.c_int]
_lib.ssa_opt_init.restype = ctypes.c_int

_lib.ssa_opt_decompose.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int]
_lib.ssa_opt_decompose.restype = ctypes.c_int

_lib.ssa_opt_decompose_block.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.ssa_opt_decompose_block.restype = ctypes.c_int

_lib.ssa_opt_decompose_randomized.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int]
_lib.ssa_opt_decompose_randomized.restype = ctypes.c_int

_lib.ssa_opt_reconstruct.argtypes = [ctypes.POINTER(_SSA_Opt), c_int_p, ctypes.c_int, c_double_p]
_lib.ssa_opt_reconstruct.restype = ctypes.c_int

_lib.ssa_opt_forecast.argtypes = [ctypes.POINTER(_SSA_Opt), c_int_p, ctypes.c_int, ctypes.c_int, c_double_p]
_lib.ssa_opt_forecast.restype = ctypes.c_int

_lib.ssa_opt_forecast_full.argtypes = [ctypes.POINTER(_SSA_Opt), c_int_p, ctypes.c_int, ctypes.c_int, c_double_p]
_lib.ssa_opt_forecast_full.restype = ctypes.c_int

_lib.ssa_opt_wcorr_matrix.argtypes = [ctypes.POINTER(_SSA_Opt), c_double_p]
_lib.ssa_opt_wcorr_matrix.restype = ctypes.c_int

_lib.ssa_opt_wcorr_pair.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int]
_lib.ssa_opt_wcorr_pair.restype = ctypes.c_double

_lib.ssa_opt_variance_explained.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int]
_lib.ssa_opt_variance_explained.restype = ctypes.c_double

_lib.ssa_opt_find_periodic_pairs.argtypes = [ctypes.POINTER(_SSA_Opt), c_int_p, ctypes.c_int, ctypes.c_double, ctypes.c_double]
_lib.ssa_opt_find_periodic_pairs.restype = ctypes.c_int

_lib.ssa_opt_get_trend.argtypes = [ctypes.POINTER(_SSA_Opt), c_double_p]
_lib.ssa_opt_get_trend.restype = ctypes.c_int

_lib.ssa_opt_get_noise.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, c_double_p]
_lib.ssa_opt_get_noise.restype = ctypes.c_int

_lib.ssa_opt_free.argtypes = [ctypes.POINTER(_SSA_Opt)]
_lib.ssa_opt_free.restype = None

# --- MSSA functions ---
_lib.mssa_opt_init.argtypes = [ctypes.POINTER(_MSSA_Opt), c_double_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.mssa_opt_init.restype = ctypes.c_int

_lib.mssa_opt_decompose.argtypes = [ctypes.POINTER(_MSSA_Opt), ctypes.c_int, ctypes.c_int]
_lib.mssa_opt_decompose.restype = ctypes.c_int

_lib.mssa_opt_reconstruct.argtypes = [ctypes.POINTER(_MSSA_Opt), ctypes.c_int, c_int_p, ctypes.c_int, c_double_p]
_lib.mssa_opt_reconstruct.restype = ctypes.c_int

_lib.mssa_opt_reconstruct_all.argtypes = [ctypes.POINTER(_MSSA_Opt), c_int_p, ctypes.c_int, c_double_p]
_lib.mssa_opt_reconstruct_all.restype = ctypes.c_int

_lib.mssa_opt_series_contributions.argtypes = [ctypes.POINTER(_MSSA_Opt), c_double_p]
_lib.mssa_opt_series_contributions.restype = ctypes.c_int

_lib.mssa_opt_variance_explained.argtypes = [ctypes.POINTER(_MSSA_Opt), ctypes.c_int, ctypes.c_int]
_lib.mssa_opt_variance_explained.restype = ctypes.c_double

_lib.mssa_opt_free.argtypes = [ctypes.POINTER(_MSSA_Opt)]
_lib.mssa_opt_free.restype = None

# ============================================================================
# Helper functions
# ============================================================================

def _to_c_array(arr: np.ndarray, dtype=np.float64) -> Tuple[ctypes.Array, int]:
    """Convert numpy array to C pointer."""
    arr = np.ascontiguousarray(arr, dtype=dtype)
    ptr = arr.ctypes.data_as(c_double_p if dtype == np.float64 else c_int_p)
    return ptr, arr

def _to_c_int_array(arr: List[int]) -> Tuple[ctypes.Array, np.ndarray]:
    """Convert list of ints to C pointer."""
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    ptr = arr.ctypes.data_as(c_int_p)
    return ptr, arr

# ============================================================================
# SSA Class (Univariate)
# ============================================================================

class SSA:
    """
    Singular Spectrum Analysis for univariate time series.
    
    Parameters
    ----------
    x : array-like
        Input time series of length N
    L : int
        Window length (embedding dimension). Typical: N//3 to N//2
    
    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 500)
    >>> x = np.sin(2*np.pi*t) + 0.5*np.random.randn(500)
    >>> 
    >>> ssa = SSA(x, L=100)
    >>> ssa.decompose(k=20)
    >>> 
    >>> # Extract trend
    >>> trend = ssa.reconstruct([0])
    >>> 
    >>> # Extract periodic component
    >>> periodic = ssa.reconstruct([1, 2])
    >>> 
    >>> # Forecast
    >>> forecast = ssa.forecast([0, 1, 2], n_forecast=50)
    """
    
    def __init__(self, x: np.ndarray, L: int):
        self._ctx = _SSA_Opt()
        self._x = np.ascontiguousarray(x, dtype=np.float64)
        self.N = len(self._x)
        self.L = L
        self.K = self.N - L + 1
        self.n_components = 0
        self._decomposed = False
        
        x_ptr = self._x.ctypes.data_as(c_double_p)
        ret = _lib.ssa_opt_init(ctypes.byref(self._ctx), x_ptr, self.N, L)
        if ret != 0:
            raise ValueError(f"SSA initialization failed (N={self.N}, L={L})")
    
    def __del__(self):
        if hasattr(self, '_ctx'):
            _lib.ssa_opt_free(ctypes.byref(self._ctx))
    
    def decompose(self, k: int, method: str = "randomized", **kwargs) -> "SSA":
        """
        Compute SVD decomposition.
        
        Parameters
        ----------
        k : int
            Number of components to compute
        method : str
            "randomized" (default, fastest), "block", or "sequential"
        **kwargs :
            - max_iter: for sequential method (default 100)
            - oversampling: for randomized method (default 8)
            - block_size: for block method (default min(k, 32))
        
        Returns
        -------
        self : for method chaining
        """
        if method == "randomized":
            oversampling = kwargs.get("oversampling", 8)
            ret = _lib.ssa_opt_decompose_randomized(ctypes.byref(self._ctx), k, oversampling)
        elif method == "block":
            block_size = kwargs.get("block_size", min(k, 32))
            max_iter = kwargs.get("max_iter", 100)
            ret = _lib.ssa_opt_decompose_block(ctypes.byref(self._ctx), k, block_size, max_iter)
        elif method == "sequential":
            max_iter = kwargs.get("max_iter", 100)
            ret = _lib.ssa_opt_decompose(ctypes.byref(self._ctx), k, max_iter)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if ret != 0:
            raise RuntimeError(f"Decomposition failed (method={method}, k={k})")
        
        self.n_components = k
        self._decomposed = True
        return self
    
    def reconstruct(self, group: List[int]) -> np.ndarray:
        """
        Reconstruct signal from selected components.
        
        Parameters
        ----------
        group : list of int
            Component indices to include
        
        Returns
        -------
        reconstructed : ndarray of shape (N,)
        """
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        output = np.zeros(self.N, dtype=np.float64)
        group_arr = np.ascontiguousarray(group, dtype=np.int32)
        
        ret = _lib.ssa_opt_reconstruct(
            ctypes.byref(self._ctx),
            group_arr.ctypes.data_as(c_int_p),
            len(group),
            output.ctypes.data_as(c_double_p)
        )
        
        if ret != 0:
            raise RuntimeError("Reconstruction failed")
        
        return output
    
    def forecast(self, group: List[int], n_forecast: int) -> np.ndarray:
        """
        Forecast future values using Linear Recurrence Formula.
        
        Parameters
        ----------
        group : list of int
            Component indices to use for forecasting
        n_forecast : int
            Number of future points to predict
        
        Returns
        -------
        forecast : ndarray of shape (n_forecast,)
        """
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        output = np.zeros(n_forecast, dtype=np.float64)
        group_arr = np.ascontiguousarray(group, dtype=np.int32)
        
        ret = _lib.ssa_opt_forecast(
            ctypes.byref(self._ctx),
            group_arr.ctypes.data_as(c_int_p),
            len(group),
            n_forecast,
            output.ctypes.data_as(c_double_p)
        )
        
        if ret != 0:
            raise RuntimeError("Forecast failed (verticality >= 1?)")
        
        return output
    
    def forecast_full(self, group: List[int], n_forecast: int) -> np.ndarray:
        """
        Get reconstruction + forecast concatenated.
        
        Returns
        -------
        full : ndarray of shape (N + n_forecast,)
        """
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        output = np.zeros(self.N + n_forecast, dtype=np.float64)
        group_arr = np.ascontiguousarray(group, dtype=np.int32)
        
        ret = _lib.ssa_opt_forecast_full(
            ctypes.byref(self._ctx),
            group_arr.ctypes.data_as(c_int_p),
            len(group),
            n_forecast,
            output.ctypes.data_as(c_double_p)
        )
        
        if ret != 0:
            raise RuntimeError("Forecast failed")
        
        return output
    
    def get_trend(self) -> np.ndarray:
        """Extract trend (component 0 only)."""
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        output = np.zeros(self.N, dtype=np.float64)
        ret = _lib.ssa_opt_get_trend(
            ctypes.byref(self._ctx),
            output.ctypes.data_as(c_double_p)
        )
        
        if ret != 0:
            raise RuntimeError("get_trend failed")
        
        return output
    
    def get_noise(self, noise_start: int) -> np.ndarray:
        """Extract noise (components from noise_start onwards)."""
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        output = np.zeros(self.N, dtype=np.float64)
        ret = _lib.ssa_opt_get_noise(
            ctypes.byref(self._ctx),
            noise_start,
            output.ctypes.data_as(c_double_p)
        )
        
        if ret != 0:
            raise RuntimeError("get_noise failed")
        
        return output
    
    def wcorr_matrix(self) -> np.ndarray:
        """
        Compute W-correlation matrix between all components.
        
        Returns
        -------
        W : ndarray of shape (n_components, n_components)
        """
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        k = self.n_components
        W = np.zeros((k, k), dtype=np.float64)
        
        ret = _lib.ssa_opt_wcorr_matrix(
            ctypes.byref(self._ctx),
            W.ctypes.data_as(c_double_p)
        )
        
        if ret != 0:
            raise RuntimeError("wcorr_matrix failed")
        
        return W
    
    def wcorr(self, i: int, j: int) -> float:
        """Compute W-correlation between two components."""
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        return _lib.ssa_opt_wcorr_pair(ctypes.byref(self._ctx), i, j)
    
    def variance_explained(self, start: int = 0, end: int = -1) -> float:
        """
        Get cumulative variance explained by component range.
        
        Parameters
        ----------
        start : int
            First component (inclusive)
        end : int
            Last component (inclusive), -1 for last
        
        Returns
        -------
        ratio : float in [0, 1]
        """
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        return _lib.ssa_opt_variance_explained(ctypes.byref(self._ctx), start, end)
    
    def find_periodic_pairs(self, max_pairs: int = 10, 
                            sv_tol: float = 0.1, 
                            wcorr_thresh: float = 0.5) -> List[Tuple[int, int]]:
        """
        Find component pairs representing periodic signals.
        
        Returns
        -------
        pairs : list of (i, j) tuples
        """
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        pairs = np.zeros(2 * max_pairs, dtype=np.int32)
        
        n_found = _lib.ssa_opt_find_periodic_pairs(
            ctypes.byref(self._ctx),
            pairs.ctypes.data_as(c_int_p),
            max_pairs,
            sv_tol,
            wcorr_thresh
        )
        
        return [(pairs[2*i], pairs[2*i+1]) for i in range(n_found)]

# ============================================================================
# MSSA Class (Multivariate)
# ============================================================================

class MSSA:
    """
    Multivariate Singular Spectrum Analysis for correlated time series.
    
    Parameters
    ----------
    X : array-like of shape (M, N)
        M time series of length N each
    L : int
        Window length (embedding dimension)
    
    Examples
    --------
    >>> import numpy as np
    >>> # 3 correlated series
    >>> t = np.linspace(0, 10, 500)
    >>> common = np.sin(2*np.pi*t)
    >>> X = np.array([
    ...     common + 0.3*np.random.randn(500),
    ...     0.8*common + 0.3*np.random.randn(500),
    ...     0.9*common + 0.3*np.random.randn(500),
    ... ])
    >>> 
    >>> mssa = MSSA(X, L=100)
    >>> mssa.decompose(k=10)
    >>> 
    >>> # Extract common factor from series 0
    >>> common_factor = mssa.reconstruct(0, [0])
    """
    
    def __init__(self, X: np.ndarray, L: int):
        self._ctx = _MSSA_Opt()
        self._X = np.ascontiguousarray(X, dtype=np.float64)
        
        if self._X.ndim != 2:
            raise ValueError("X must be 2D array of shape (M, N)")
        
        self.M, self.N = self._X.shape
        self.L = L
        self.K = self.N - L + 1
        self.n_components = 0
        self._decomposed = False
        
        X_ptr = self._X.ctypes.data_as(c_double_p)
        ret = _lib.mssa_opt_init(ctypes.byref(self._ctx), X_ptr, self.M, self.N, L)
        if ret != 0:
            raise ValueError(f"MSSA initialization failed (M={self.M}, N={self.N}, L={L})")
    
    def __del__(self):
        if hasattr(self, '_ctx'):
            _lib.mssa_opt_free(ctypes.byref(self._ctx))
    
    def decompose(self, k: int, oversampling: int = 8) -> "MSSA":
        """
        Compute joint SVD decomposition.
        
        Parameters
        ----------
        k : int
            Number of components to compute
        oversampling : int
            Oversampling for randomized SVD (default 8)
        
        Returns
        -------
        self : for method chaining
        """
        ret = _lib.mssa_opt_decompose(ctypes.byref(self._ctx), k, oversampling)
        if ret != 0:
            raise RuntimeError(f"MSSA decomposition failed (k={k})")
        
        self.n_components = k
        self._decomposed = True
        return self
    
    def reconstruct(self, series_idx: int, group: List[int]) -> np.ndarray:
        """
        Reconstruct a single series from selected components.
        
        Parameters
        ----------
        series_idx : int
            Which series to reconstruct (0 to M-1)
        group : list of int
            Component indices to include
        
        Returns
        -------
        reconstructed : ndarray of shape (N,)
        """
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        output = np.zeros(self.N, dtype=np.float64)
        group_arr = np.ascontiguousarray(group, dtype=np.int32)
        
        ret = _lib.mssa_opt_reconstruct(
            ctypes.byref(self._ctx),
            series_idx,
            group_arr.ctypes.data_as(c_int_p),
            len(group),
            output.ctypes.data_as(c_double_p)
        )
        
        if ret != 0:
            raise RuntimeError("Reconstruction failed")
        
        return output
    
    def reconstruct_all(self, group: List[int]) -> np.ndarray:
        """
        Reconstruct all series from selected components.
        
        Returns
        -------
        reconstructed : ndarray of shape (M, N)
        """
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        output = np.zeros((self.M, self.N), dtype=np.float64)
        group_arr = np.ascontiguousarray(group, dtype=np.int32)
        
        ret = _lib.mssa_opt_reconstruct_all(
            ctypes.byref(self._ctx),
            group_arr.ctypes.data_as(c_int_p),
            len(group),
            output.ctypes.data_as(c_double_p)
        )
        
        if ret != 0:
            raise RuntimeError("Reconstruction failed")
        
        return output
    
    def series_contributions(self) -> np.ndarray:
        """
        Get contribution of each series to each component.
        
        Returns
        -------
        contributions : ndarray of shape (M, k)
            contributions[m, i] = fraction of component i's energy from series m
        """
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        contrib = np.zeros((self.M, self.n_components), dtype=np.float64)
        
        ret = _lib.mssa_opt_series_contributions(
            ctypes.byref(self._ctx),
            contrib.ctypes.data_as(c_double_p)
        )
        
        if ret != 0:
            raise RuntimeError("series_contributions failed")
        
        return contrib
    
    def variance_explained(self, start: int = 0, end: int = -1) -> float:
        """Get cumulative variance explained by component range."""
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        return _lib.mssa_opt_variance_explained(ctypes.byref(self._ctx), start, end)


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    print("Testing SSA wrapper...")
    
    # Generate test signal
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    signal = 10 + 0.1*t + 2*np.sin(2*np.pi*t) + 0.5*np.random.randn(500)
    
    # Test SSA
    ssa = SSA(signal, L=100)
    ssa.decompose(k=20)
    
    print(f"  Variance explained (first 3): {ssa.variance_explained(0, 2):.2%}")
    
    trend = ssa.get_trend()
    print(f"  Trend extracted: shape={trend.shape}")
    
    forecast = ssa.forecast([0, 1, 2], n_forecast=50)
    print(f"  Forecast: shape={forecast.shape}")
    
    pairs = ssa.find_periodic_pairs()
    print(f"  Periodic pairs found: {pairs}")
    
    # Test MSSA
    X = np.array([signal, 0.9*signal + 0.3*np.random.randn(500)])
    mssa = MSSA(X, L=100)
    mssa.decompose(k=10)
    
    print(f"  MSSA variance explained (first 3): {mssa.variance_explained(0, 2):.2%}")
    
    contrib = mssa.series_contributions()
    print(f"  Series contributions shape: {contrib.shape}")
    
    print("\nAll tests passed!")