"""
SSA Python Wrapper
==================

Ctypes wrapper for ssa_opt.h MKL-optimized SSA implementation.

Usage:
    from ssa_wrapper import SSA, MSSA, cadzow, CadzowResult
    
    # Univariate SSA
    ssa = SSA(signal, L=100)
    ssa.decompose(k=20)
    trend = ssa.reconstruct([0])
    forecast = ssa.forecast([0, 1, 2], n_forecast=50)
    vforecast = ssa.vforecast([0, 1, 2], n_forecast=50)  # Alternative forecast
    
    # Cadzow denoising
    result = cadzow(noisy_signal, L=100, rank=6)
    clean = result.signal
    
    # Or via SSA object
    result = ssa.cadzow(rank=6)
    
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

class _SSA_CadzowResult(ctypes.Structure):
    """Result struct for Cadzow iterations."""
    _fields_ = [
        ("iterations", ctypes.c_int),
        ("final_diff", ctypes.c_double),
        ("converged", ctypes.c_double)
    ]

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

# Malloc-free hot path functions
_lib.ssa_opt_prepare.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int]
_lib.ssa_opt_prepare.restype = ctypes.c_int

_lib.ssa_opt_update_signal.argtypes = [ctypes.POINTER(_SSA_Opt), c_double_p]
_lib.ssa_opt_update_signal.restype = ctypes.c_int

_lib.ssa_opt_free_prepared.argtypes = [ctypes.POINTER(_SSA_Opt)]
_lib.ssa_opt_free_prepared.restype = None

_lib.ssa_opt_reconstruct.argtypes = [ctypes.POINTER(_SSA_Opt), c_int_p, ctypes.c_int, c_double_p]
_lib.ssa_opt_reconstruct.restype = ctypes.c_int

_lib.ssa_opt_forecast.argtypes = [ctypes.POINTER(_SSA_Opt), c_int_p, ctypes.c_int, ctypes.c_int, c_double_p]
_lib.ssa_opt_forecast.restype = ctypes.c_int

_lib.ssa_opt_forecast_full.argtypes = [ctypes.POINTER(_SSA_Opt), c_int_p, ctypes.c_int, ctypes.c_int, c_double_p]
_lib.ssa_opt_forecast_full.restype = ctypes.c_int

# Vector forecast (V-forecast) - alternative to recurrent forecast
# These are optional - will fail gracefully if DLL not rebuilt
_HAS_VFORECAST = False
try:
    _lib.ssa_opt_vforecast.argtypes = [ctypes.POINTER(_SSA_Opt), c_int_p, ctypes.c_int, ctypes.c_int, c_double_p]
    _lib.ssa_opt_vforecast.restype = ctypes.c_int
    _lib.ssa_opt_vforecast_full.argtypes = [ctypes.POINTER(_SSA_Opt), c_int_p, ctypes.c_int, ctypes.c_int, c_double_p]
    _lib.ssa_opt_vforecast_full.restype = ctypes.c_int
    _lib.ssa_opt_vforecast_fast.argtypes = [ctypes.POINTER(_SSA_Opt), c_int_p, ctypes.c_int, c_double_p, ctypes.c_int, ctypes.c_int, c_double_p]
    _lib.ssa_opt_vforecast_fast.restype = ctypes.c_int
    _HAS_VFORECAST = True
except AttributeError:
    pass  # DLL doesn't have vforecast yet

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

# Getters for decomposition results
_HAS_GETTERS = False
try:
    _lib.ssa_opt_get_singular_values.argtypes = [ctypes.POINTER(_SSA_Opt), c_double_p, ctypes.c_int]
    _lib.ssa_opt_get_singular_values.restype = ctypes.c_int
    
    _lib.ssa_opt_get_eigenvalues.argtypes = [ctypes.POINTER(_SSA_Opt), c_double_p, ctypes.c_int]
    _lib.ssa_opt_get_eigenvalues.restype = ctypes.c_int
    
    _lib.ssa_opt_get_total_variance.argtypes = [ctypes.POINTER(_SSA_Opt)]
    _lib.ssa_opt_get_total_variance.restype = ctypes.c_double
    _HAS_GETTERS = True
except AttributeError:
    pass  # DLL doesn't have getters yet

_lib.ssa_opt_free.argtypes = [ctypes.POINTER(_SSA_Opt)]
_lib.ssa_opt_free.restype = None

# --- Cadzow iterations (optional - requires rebuilt DLL) ---
_HAS_CADZOW = False
try:
    _lib.ssa_opt_cadzow.argtypes = [c_double_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                    ctypes.c_int, ctypes.c_double, c_double_p, 
                                    ctypes.POINTER(_SSA_CadzowResult)]
    _lib.ssa_opt_cadzow.restype = ctypes.c_int
    
    _lib.ssa_opt_cadzow_weighted.argtypes = [c_double_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_double, ctypes.c_double, 
                                             c_double_p, ctypes.POINTER(_SSA_CadzowResult)]
    _lib.ssa_opt_cadzow_weighted.restype = ctypes.c_int
    
    _lib.ssa_opt_cadzow_inplace.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int, 
                                            ctypes.c_double, ctypes.POINTER(_SSA_CadzowResult)]
    _lib.ssa_opt_cadzow_inplace.restype = ctypes.c_int
    _HAS_CADZOW = True
except AttributeError:
    pass  # DLL doesn't have cadzow yet

# --- ESPRIT (parestimate) ---
class _SSA_ParEstimate(ctypes.Structure):
    _fields_ = [
        ("periods", c_double_p),
        ("frequencies", c_double_p),
        ("moduli", c_double_p),
        ("rates", c_double_p),
        ("n_components", ctypes.c_int)
    ]

_HAS_ESPRIT = False
try:
    _lib.ssa_opt_parestimate.argtypes = [ctypes.POINTER(_SSA_Opt), c_int_p, ctypes.c_int, 
                                          ctypes.POINTER(_SSA_ParEstimate)]
    _lib.ssa_opt_parestimate.restype = ctypes.c_int
    
    _lib.ssa_opt_parestimate_free.argtypes = [ctypes.POINTER(_SSA_ParEstimate)]
    _lib.ssa_opt_parestimate_free.restype = None
    _HAS_ESPRIT = True
except AttributeError:
    pass  # DLL doesn't have ESPRIT yet

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
# Cadzow Iterations (Standalone Function)
# ============================================================================

class CadzowResult:
    """Result of Cadzow iterations."""
    def __init__(self, signal: np.ndarray, iterations: int, final_diff: float, converged: bool):
        self.signal = signal
        self.iterations = iterations
        self.final_diff = final_diff
        self.converged = converged
    
    def __repr__(self):
        return (f"CadzowResult(iterations={self.iterations}, "
                f"final_diff={self.final_diff:.2e}, converged={self.converged})")


class ParEstimate:
    """
    Result of ESPRIT frequency estimation.
    
    Attributes
    ----------
    periods : ndarray
        Estimated periods in samples. Inf for DC/trend components.
    frequencies : ndarray  
        Frequencies in cycles per sample (0 to 0.5).
    moduli : ndarray
        Eigenvalue moduli (damping factor). 1.0 = pure undamped sinusoid.
    rates : ndarray
        Damping rates = log(modulus). 0 = undamped, negative = decaying.
    n_components : int
        Number of components analyzed.
    """
    def __init__(self, periods: np.ndarray, frequencies: np.ndarray, 
                 moduli: np.ndarray, rates: np.ndarray):
        self.periods = periods
        self.frequencies = frequencies
        self.moduli = moduli
        self.rates = rates
        self.n_components = len(periods)
    
    def __repr__(self):
        return f"ParEstimate(n_components={self.n_components})"
    
    def summary(self) -> str:
        """Return formatted summary of detected components."""
        lines = ["ESPRIT Frequency Estimation Results", "=" * 50]
        lines.append(f"{'Comp':<6} {'Period':<12} {'Frequency':<12} {'Modulus':<10} {'Rate':<10}")
        lines.append("-" * 50)
        
        # Sort by period (descending, so trends first)
        order = np.argsort(-self.periods)
        
        for i, idx in enumerate(order):
            period = self.periods[idx]
            freq = self.frequencies[idx]
            mod = self.moduli[idx]
            rate = self.rates[idx]
            
            if np.isinf(period):
                period_str = "Inf (trend)"
            else:
                period_str = f"{period:.2f}"
            
            lines.append(f"{idx:<6} {period_str:<12} {freq:<12.6f} {mod:<10.4f} {rate:<10.4f}")
        
        return "\n".join(lines)
    
    def get_periodic_components(self, min_period: float = 2.0, max_period: float = None,
                                 min_modulus: float = 0.9) -> List[int]:
        """
        Get indices of components that appear to be periodic (undamped sinusoids).
        
        Parameters
        ----------
        min_period : float
            Minimum period to consider (default 2 = Nyquist limit)
        max_period : float
            Maximum period to consider (default None = no limit)
        min_modulus : float
            Minimum modulus to consider "undamped" (default 0.9)
        
        Returns
        -------
        indices : list of int
            Component indices that appear periodic
        """
        indices = []
        for i in range(self.n_components):
            period = self.periods[i]
            mod = self.moduli[i]
            
            if np.isinf(period):
                continue  # Skip trend
            if period < min_period:
                continue  # Below Nyquist
            if max_period is not None and period > max_period:
                continue
            if mod < min_modulus:
                continue  # Too damped
            
            indices.append(i)
        
        return indices

def cadzow(x: np.ndarray, L: int, rank: int, max_iter: int = 20, tol: float = 1e-9,
           alpha: float = 1.0) -> CadzowResult:
    """
    Cadzow iterations for finite-rank signal approximation.
    
    Iteratively projects signal onto the space of signals with exactly
    rank-r trajectory matrices. Converges to a signal that is exactly
    representable as a sum of `rank` exponentials/sinusoids.
    
    Parameters
    ----------
    x : ndarray
        Input signal of length N
    L : int  
        Window length (embedding dimension)
    rank : int
        Target rank (number of components to keep)
    max_iter : int, default 20
        Maximum number of iterations
    tol : float, default 1e-9
        Convergence tolerance (relative change in signal)
    alpha : float, default 1.0
        Blending parameter: output = alpha * cadzow + (1-alpha) * original
        Use alpha < 1 for regularization (less aggressive denoising)
    
    Returns
    -------
    result : CadzowResult
        - signal: denoised signal
        - iterations: number of iterations performed
        - final_diff: final relative difference
        - converged: True if converged before max_iter
    
    Examples
    --------
    >>> # Simple denoising
    >>> x_noisy = signal + noise
    >>> result = cadzow(x_noisy, L=100, rank=6)
    >>> x_clean = result.signal
    
    >>> # Conservative denoising with blending
    >>> result = cadzow(x_noisy, L=100, rank=6, alpha=0.8)
    """
    if not _HAS_CADZOW:
        raise RuntimeError("Cadzow not available - rebuild DLL with updated ssa_opt.h")
    
    x = np.ascontiguousarray(x, dtype=np.float64)
    N = len(x)
    output = np.zeros(N, dtype=np.float64)
    c_result = _SSA_CadzowResult()
    
    if alpha == 1.0:
        ret = _lib.ssa_opt_cadzow(
            x.ctypes.data_as(c_double_p),
            N, L, rank, max_iter, tol,
            output.ctypes.data_as(c_double_p),
            ctypes.byref(c_result)
        )
    else:
        ret = _lib.ssa_opt_cadzow_weighted(
            x.ctypes.data_as(c_double_p),
            N, L, rank, max_iter, tol, alpha,
            output.ctypes.data_as(c_double_p),
            ctypes.byref(c_result)
        )
    
    if ret < 0:
        raise RuntimeError("Cadzow iterations failed")
    
    return CadzowResult(
        signal=output,
        iterations=c_result.iterations,
        final_diff=c_result.final_diff,
        converged=bool(c_result.converged > 0.5)
    )

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
        self._prepared = False
        self._prepared_k = 0
        self._prepared_oversampling = 0
        
        x_ptr = self._x.ctypes.data_as(c_double_p)
        ret = _lib.ssa_opt_init(ctypes.byref(self._ctx), x_ptr, self.N, L)
        if ret != 0:
            raise ValueError(f"SSA initialization failed (N={self.N}, L={L})")
    
    def __del__(self):
        if hasattr(self, '_ctx'):
            _lib.ssa_opt_free(ctypes.byref(self._ctx))
    
    def prepare(self, max_k: int, oversampling: int = 8) -> "SSA":
        """
        Pre-allocate workspace for malloc-free hot path.
        
        Call this once before a streaming loop to eliminate allocations
        from decompose_randomized(). Workspace supports k <= max_k.
        
        Parameters
        ----------
        max_k : int
            Maximum number of components to support
        oversampling : int
            Oversampling parameter (default 8)
        
        Returns
        -------
        self : for method chaining
        """
        ret = _lib.ssa_opt_prepare(ctypes.byref(self._ctx), max_k, oversampling)
        if ret != 0:
            raise RuntimeError(f"prepare() failed (max_k={max_k})")
        self._prepared = True
        self._prepared_k = max_k
        self._prepared_oversampling = oversampling
        return self
    
    def update_signal(self, new_x: np.ndarray) -> "SSA":
        """
        Update signal data without reallocation.
        
        For streaming applications: just memcpy + 1 FFT, no malloc.
        
        Parameters
        ----------
        new_x : array-like
            New signal data (must be same length as original)
        
        Returns
        -------
        self : for method chaining
        """
        new_x = np.ascontiguousarray(new_x, dtype=np.float64)
        if len(new_x) != self.N:
            raise ValueError(f"Signal length mismatch: {len(new_x)} vs {self.N}")
        
        x_ptr = new_x.ctypes.data_as(c_double_p)
        ret = _lib.ssa_opt_update_signal(ctypes.byref(self._ctx), x_ptr)
        if ret != 0:
            raise RuntimeError("update_signal() failed")
        
        self._x = new_x
        return self
    
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
            
            # Auto-prepare if not already done or if k+oversampling exceeds prepared size
            kp = k + oversampling
            if not self._prepared or kp > self._prepared_k + self._prepared_oversampling:
                self.prepare(k, oversampling)
            
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
    
    @property
    def singular_values(self) -> np.ndarray:
        """Get singular values (σ) from decomposition."""
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        if not _HAS_GETTERS:
            raise RuntimeError("Getters not available - rebuild DLL with updated ssa_opt.h")
        output = np.zeros(self.n_components, dtype=np.float64)
        n = _lib.ssa_opt_get_singular_values(
            ctypes.byref(self._ctx),
            output.ctypes.data_as(c_double_p),
            self.n_components
        )
        return output[:n]
    
    @property
    def eigenvalues(self) -> np.ndarray:
        """Get eigenvalues (σ²) from decomposition."""
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        if not _HAS_GETTERS:
            raise RuntimeError("Getters not available - rebuild DLL with updated ssa_opt.h")
        output = np.zeros(self.n_components, dtype=np.float64)
        n = _lib.ssa_opt_get_eigenvalues(
            ctypes.byref(self._ctx),
            output.ctypes.data_as(c_double_p),
            self.n_components
        )
        return output[:n]
    
    @property
    def total_variance(self) -> float:
        """Get total variance captured by all computed components."""
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        if not _HAS_GETTERS:
            raise RuntimeError("Getters not available - rebuild DLL with updated ssa_opt.h")
        return _lib.ssa_opt_get_total_variance(ctypes.byref(self._ctx))
    
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
    
    def vforecast(self, group: List[int], n_forecast: int) -> np.ndarray:
        """
        Vector forecast (V-forecast) - alternative to recurrent forecast.
        
        Projects onto eigenvector subspace at each step instead of using
        precomputed LRR coefficients. Can be more numerically stable for
        long forecast horizons.
        
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
        if not _HAS_VFORECAST:
            raise RuntimeError("vforecast not available - rebuild DLL with updated ssa_opt.h")
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        output = np.zeros(n_forecast, dtype=np.float64)
        group_arr = np.ascontiguousarray(group, dtype=np.int32)
        
        ret = _lib.ssa_opt_vforecast(
            ctypes.byref(self._ctx),
            group_arr.ctypes.data_as(c_int_p),
            len(group),
            n_forecast,
            output.ctypes.data_as(c_double_p)
        )
        
        if ret != 0:
            raise RuntimeError("V-forecast failed (verticality >= 1?)")
        
        return output
    
    def vforecast_full(self, group: List[int], n_forecast: int) -> np.ndarray:
        """
        Get reconstruction + V-forecast concatenated.
        
        Returns
        -------
        full : ndarray of shape (N + n_forecast,)
        """
        if not _HAS_VFORECAST:
            raise RuntimeError("vforecast not available - rebuild DLL with updated ssa_opt.h")
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        output = np.zeros(self.N + n_forecast, dtype=np.float64)
        group_arr = np.ascontiguousarray(group, dtype=np.int32)
        
        ret = _lib.ssa_opt_vforecast_full(
            ctypes.byref(self._ctx),
            group_arr.ctypes.data_as(c_int_p),
            len(group),
            n_forecast,
            output.ctypes.data_as(c_double_p)
        )
        
        if ret != 0:
            raise RuntimeError("V-forecast failed")
        
        return output
    
    def vforecast_fast(self, group: List[int], base_signal: np.ndarray, n_forecast: int) -> np.ndarray:
        """
        Fast V-forecast from arbitrary base signal (for hot loops).
        
        Uses BLAS for inner products. Does not require reconstruction -
        you provide the base signal directly.
        
        Parameters
        ----------
        group : list of int
            Component indices to use for forecasting
        base_signal : ndarray
            Signal to forecast from (must be at least L-1 long)
        n_forecast : int
            Number of future points to predict
        
        Returns
        -------
        forecast : ndarray of shape (n_forecast,)
        """
        if not _HAS_VFORECAST:
            raise RuntimeError("vforecast not available - rebuild DLL with updated ssa_opt.h")
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        base = np.ascontiguousarray(base_signal, dtype=np.float64)
        output = np.zeros(n_forecast, dtype=np.float64)
        group_arr = np.ascontiguousarray(group, dtype=np.int32)
        
        ret = _lib.ssa_opt_vforecast_fast(
            ctypes.byref(self._ctx),
            group_arr.ctypes.data_as(c_int_p),
            len(group),
            base.ctypes.data_as(c_double_p),
            len(base),
            n_forecast,
            output.ctypes.data_as(c_double_p)
        )
        
        if ret != 0:
            raise RuntimeError("V-forecast fast failed")
        
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
    
    def cadzow(self, rank: int, max_iter: int = 20, tol: float = 1e-9, 
               alpha: float = 1.0) -> 'CadzowResult':
        """
        Apply Cadzow iterations for finite-rank signal approximation.
        
        Uses this SSA object's signal and window length. Returns denoised
        signal whose trajectory matrix is exactly rank-r.
        
        Parameters
        ----------
        rank : int
            Target rank (number of components to keep)
        max_iter : int, default 20
            Maximum number of iterations
        tol : float, default 1e-9
            Convergence tolerance (relative change in signal)
        alpha : float, default 1.0
            Blending: output = alpha * cadzow + (1-alpha) * original
        
        Returns
        -------
        result : CadzowResult
            - signal: denoised signal
            - iterations: number performed
            - final_diff: final relative difference  
            - converged: True if converged
        
        Examples
        --------
        >>> ssa = SSA(noisy_signal, L=250)
        >>> result = ssa.cadzow(rank=6)
        >>> clean_signal = result.signal
        """
        return cadzow(self._x, self.L, rank, max_iter, tol, alpha)
    
    def parestimate(self, group: List[int] = None) -> 'ParEstimate':
        """
        Estimate periods/frequencies using ESPRIT method.
        
        Extracts dominant periods from the eigenvectors of the selected
        components. Useful for detecting market cycles and optimal L selection.
        
        Parameters
        ----------
        group : list of int, optional
            Component indices to analyze. If None, uses all decomposed components.
        
        Returns
        -------
        result : ParEstimate
            - periods: Estimated periods in samples
            - frequencies: Frequencies in cycles per sample
            - moduli: Eigenvalue moduli (1.0 = undamped sinusoid)
            - rates: Damping rates (0 = undamped)
        
        Examples
        --------
        >>> ssa = SSA(prices, L=120)
        >>> ssa.decompose(k=10)
        >>> par = ssa.parestimate()
        >>> print(par.summary())
        >>> 
        >>> # Find dominant cycles
        >>> periodic = par.get_periodic_components(min_period=5, min_modulus=0.95)
        >>> print(f"Dominant periods: {par.periods[periodic]}")
        """
        if not _HAS_ESPRIT:
            raise RuntimeError("ESPRIT not available - rebuild DLL with updated ssa_opt.h")
        if not self._decomposed:
            raise RuntimeError("Call decompose() first")
        
        c_result = _SSA_ParEstimate()
        
        if group is not None and len(group) > 0:
            group_arr = np.ascontiguousarray(group, dtype=np.int32)
            ret = _lib.ssa_opt_parestimate(
                ctypes.byref(self._ctx),
                group_arr.ctypes.data_as(c_int_p),
                len(group),
                ctypes.byref(c_result)
            )
        else:
            ret = _lib.ssa_opt_parestimate(
                ctypes.byref(self._ctx),
                None,
                0,
                ctypes.byref(c_result)
            )
        
        if ret != 0:
            raise RuntimeError("parestimate failed")
        
        # Copy data to numpy arrays before freeing
        n = c_result.n_components
        periods = np.ctypeslib.as_array(c_result.periods, shape=(n,)).copy()
        frequencies = np.ctypeslib.as_array(c_result.frequencies, shape=(n,)).copy()
        moduli = np.ctypeslib.as_array(c_result.moduli, shape=(n,)).copy()
        rates = np.ctypeslib.as_array(c_result.rates, shape=(n,)).copy()
        
        # Free C memory
        _lib.ssa_opt_parestimate_free(ctypes.byref(c_result))
        
        return ParEstimate(periods, frequencies, moduli, rates)
    
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
    
    # Test streaming update
    print("\n  Testing streaming updates...")
    new_signal = signal + 0.1 * np.random.randn(500)
    ssa.update_signal(new_signal)
    ssa.decompose(k=20)
    print(f"  After update - variance explained: {ssa.variance_explained(0, 2):.2%}")
    
    # Test MSSA
    X = np.array([signal, 0.9*signal + 0.3*np.random.randn(500)])
    mssa = MSSA(X, L=100)
    mssa.decompose(k=10)
    
    print(f"\n  MSSA variance explained (first 3): {mssa.variance_explained(0, 2):.2%}")
    
    contrib = mssa.series_contributions()
    print(f"  Series contributions shape: {contrib.shape}")
    
    print("\nAll tests passed!")