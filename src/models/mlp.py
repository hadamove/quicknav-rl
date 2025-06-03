
import flax.linen as nn

import jax.numpy as jnp



class MLP(nn.Module):
    hidden_size: list[int]
    dropout_rate: float = 0.5
    deterministic: bool = True
    mem_len: int = 128

    def separate_inputs(self, inputs, batch_size):
        memory = inputs[:, -self.mem_len:]
        x = inputs[:, :-self.mem_len]
        memory = jnp.reshape(memory, (-1, self.mem_len))

        return  jnp.reshape(x, (batch_size, -1)), memory

    @nn.compact
    def __call__(self, inpt):

        # print("INPT", inpt, flush=True)

        # x = inpt[:, :-128]
        # memory = inpt[:, -128:]

        # print(x, flush=True)
        x = inpt

        for features in self.hidden_size:
            x = nn.Dense(features)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)

        action = nn.Dense(2)(x).squeeze()
        memory = nn.Dense(self.mem_len)(x)
        

        return action, memory # memory
    
    @staticmethod
    def initialize_state(mem_len):
        return jnp.zeros(mem_len)