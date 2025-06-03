import jax.numpy as jnp
import jax.nn as nn
import jax

from flax import linen as nn
from flax.linen.initializers import constant, orthogonal



def tree_index(tree,i):
    """Stack tree or a 

    Args:
        tree (_type_): _description_
        i (_type_): _description_

    Returns:
        _type_: _description_
    """
    return jax.tree_map(lambda x:x[i] ,tree)



class LSTM(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, inputs, last_state):
        d_model = self.d_model

        class LSTMout(nn.Module):
            @nn.compact    
            def __call__(self, carry, inputs):
                return nn.OptimizedLSTMCell(
                    features=d_model,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    recurrent_kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0)
                )(carry,inputs)

        model = nn.scan(
            LSTMout, variable_broadcast="params", split_rngs={"params": False},)
        
        return model()((last_state[0], last_state[1]), inputs)
    
    def initialize_state(self):
         return (
          jnp.zeros((self.d_model,)),  # c
          jnp.zeros((self.d_model,))   # h
      )



class LSTMMultiLayer(nn.Module):
    d_model: int
    n_layers: int

    def separate_inputs(self, inputs, batch_size):
        mem_len = self.n_layers * self.d_model * 2
        memory = inputs[:, -mem_len:]

        x = inputs[:, :-mem_len]
        memory = jnp.reshape(memory, (self.n_layers, 2, batch_size, self.d_model))

        return  jnp.reshape(x, (1, batch_size, -1)), memory

    @nn.compact
    def __call__(self, inputs):
        x, last_memory = self.separate_inputs(inputs, inputs.shape[0])
        new_memory=[None]*self.n_layers

        for i in range(self.n_layers):
            carry, x = LSTM(self.d_model)(x, last_memory[i])
            new_memory[i] = jnp.concatenate(carry, axis=-1)

        new_memory= jnp.concatenate(new_memory, axis=-1).reshape(-1, self.n_layers * self.d_model * 2)

        x = nn.Dense(2)(x)

        return x.squeeze(), new_memory.squeeze()
    
    @staticmethod
    def initialize_state(d_model, n_layers):
        return jnp.concatenate([jnp.zeros((2 * d_model,)) for _ in range(n_layers)]).reshape(-1).squeeze()

