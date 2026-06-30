import numpy as np
from scipy.stats import norm

class Uniform:
    
    """
    Uniform prior on [lower, upper] with ppf mapping from [0,1] to parameter space.
    
    Parameters
    ----------
    lower : float
        Lower bound.
    upper : float
        Upper bound.
    """
    
    def __init__(self, lower=0.0, upper=1.0):
        self.lower = lower
        self.upper = upper

    def ppf(self, x):
        return x * self.upper + self.lower
        
class LogUniform:
    """
    Log-uniform prior ( log10(theta) ~ Uniform(log10(lower), log10(upper)) )
    with ppf mapping from [0,1] to parameter space.
    
    Parameters
    ----------
    lower : float
        Lower bound.
    upper : float
        Upper bound.

    """
    def __init__(self, lower, upper):
        assert lower > 0
        assert upper > lower
        self.lower = lower
        self.upper = upper
        self.logL = np.log10(lower)
        self.logU = np.log10(upper)

    def ppf(self, x):
        log10_theta = self.logL + x * (self.logU - self.logL)
        return 10.0 ** log10_theta

class SignedLogUniform:
    
    """
    Signed log-uniform prior on [-upper, -lower] or [lower, upper]
    with ppf mapping from [0,1] to parameter space.
    As LogUniform, but reflected around 0.

    Parameters
    ----------
    lower : float
        Lower bound on |b| (must be > 0).
    upper : float
        Upper bound on |b|.
    """

    def __init__(self, lower, upper, sign):
        assert lower > 0
        assert upper > lower
        self.b_min = lower
        self.b_max = upper
        self.log_b_min = np.log(lower)
        self.log_b_max = np.log(upper)
        self.sign = sign

    def ppf(self, x):

        # Log-uniform magnitude
        log_abs_b = self.log_b_min + x * (self.log_b_max - self.log_b_min)
        abs_b = 10 ** log_abs_b

        return self.sign * abs_b

class Normal:
    
    """
    Normal (Gaussian) prior with ppf mapping from [0,1] to parameter space.
    
    Parameters
    ----------
    mean : float
        Mean of the normal distribution.
    std : float
        Standard deviation.
    lower : float or None
        Lower truncation bound. If None, no lower truncation.
    upper : float or None
        Upper truncation bound. If None, no upper truncation.
    """
    
    def __init__(self, mean, std, lower=None, upper=None):
        assert std > 0
        self.mean = mean
        self.std = std
        self.lower = lower
        self.upper = upper

        # Convert bounds to standardized normal units
        z_lower = (lower - mean) / std if lower is not None else -np.inf
        z_upper = (upper - mean) / std if upper is not None else np.inf

        # Compute CDF at truncation bounds
        self.cdf_L = norm.cdf(z_lower)
        self.cdf_U = norm.cdf(z_upper)

        # Normalization constant
        self.Z = self.cdf_U - self.cdf_L

    def ppf(self, x):
        # Map x into truncated CDF range
        y = self.cdf_L + x * self.Z
        
        # Invert standard normal CDF
        z = norm.ppf(y)
        
        # Convert back to normal scale
        return self.mean + self.std * z


class Log10Normal:
    """
    LogNormal prior where log10(theta) ~ Normal(meanlog10, stdlog10),
    with optional truncation in linear space.

    Parameters
    ----------
    meanlog10 : float
        Mean of log10(theta).
    stdlog10 : float
        Std deviation of log10(theta).
    lower : float or None
        Lower truncation bound in linear space (must be > 0).
        If None, no lower truncation.
    upper : float or None
        Upper truncation bound in linear space.
        If None, no upper truncation.
    """
    def __init__(self, meanlog10, stdlog10, lower=None, upper=None):
        assert stdlog10 > 0
        self.meanlog10 = meanlog10
        self.stdlog10 = stdlog10

        # Handle truncation
        self.lower = lower
        self.upper = upper

        # Convert truncation limits to log10-space
        self.log10L = np.log10(lower) if (lower is not None and lower > 0) else -np.inf
        self.log10U = np.log10(upper) if (upper is not None) else np.inf

        # Convert to standardized normal coordinates
        zL = (self.log10L - self.meanlog10) / self.stdlog10
        zU = (self.log10U - self.meanlog10) / self.stdlog10

        # CDF bounds for truncation
        self.cdf_L = norm.cdf(zL)
        self.cdf_U = norm.cdf(zU)

        # Normalization constant
        self.Z = self.cdf_U - self.cdf_L

        if self.Z <= 0:
            raise ValueError("Invalid truncation bounds: zero prior mass.")

    def ppf(self, x):
        """
        Transform a unit-uniform sample x ∈ [0,1] into a LogNormal10 sample.
        """
        # Map x into truncated CDF range
        y = self.cdf_L + x * self.Z

        # Invert normal CDF in log10-space
        log10_theta = self.meanlog10 + self.stdlog10 * norm.ppf(y)

        # Convert back to linear space
        return 10.0 ** log10_theta