def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	z=max(alpha*z,z)
	return z
	# Your code here
	pass
