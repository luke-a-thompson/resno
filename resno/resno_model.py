import jax
import equinox as eqx


class RESNOBlock(eqx.Module):
    linear_in: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    rmsnorm: eqx.nn.RMSNorm

    def __init__(self, in_sig_dim: int, out_sig_dim: int, key: jax.Array) -> None:
        super().__init__()
        self.linear_in = eqx.nn.Linear(in_sig_dim, out_sig_dim, key=key)
        self.rmsnorm = eqx.nn.RMSNorm((out_sig_dim,))
        self.linear_out = eqx.nn.Linear(out_sig_dim, in_sig_dim, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear_in(x)
        x = jax.nn.gelu(x)
        x = self.rmsnorm(x)
        x = self.linear_out(x)
        return x


class SignatureInverter(eqx.Module):
    lin: eqx.nn.Linear

    def __init__(self, in_sig_dim: int, out_path_dim: int, key: jax.Array) -> None:
        super().__init__()
        self.lin = eqx.nn.Linear(in_sig_dim, out_path_dim, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.lin(x)
        return x


class RESNO(eqx.Module):
    resno_blocks: list[RESNOBlock]

    signature_inverter: SignatureInverter

    def __init__(self, in_sig_dim: int, out_sig_dim: int, out_path_dim: int, num_resno_blocks: int, key: jax.Array) -> None:
        super().__init__()
        self.resno_blocks = [RESNOBlock(in_sig_dim, out_sig_dim, key=key) for _ in range(num_resno_blocks)]

        self.signature_inverter = SignatureInverter(out_sig_dim, out_path_dim, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        for block in self.resno_blocks:
            x = block(x)

        x = self.signature_inverter(x)
        return x
