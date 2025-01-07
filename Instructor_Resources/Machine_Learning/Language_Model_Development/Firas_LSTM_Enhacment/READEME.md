# LSTM Enhancement Cloud Recording


Language modelling is my playing field and I wanted to make sure you guys go off with best practices, at least as a starter, from building LSTM models, to actually modelling sequences. Keep in mind that the model and input/output change depending on your task and objective off model usage.

```python
How RNNs work under the hood:
    xs = np.zeros((len(unique_tokens),1))
    h_prev = np.zeros((10,1))
    Wxh = np.random.randn(10, len(unique_tokens))*0.01 # input to hidden
    Whh = np.random.randn(10, 10)*0.01 # hidden to hidden
    Why = np.random.randn(len(uniqye_characters), 10)*0.01 # hidden to output
    bh = np.zeros((10, 1)) # hidden bias
    by = np.zeros((len(uniqye_characters), 1)) # output bias
    hs = np.tanh(np.dot(Wxh, xs) + np.dot(Whh, h_prev) + bh) # hidden state
    y = np.softmax(np.dot(W_hy, hs))
    loss = -np.log(y[range(len(y)), y_target])
```

Please find the recording I just created and the up to par LSTM sequence 2 sequence language modelling. I found certain nuances in the curriculums steps, given I have went through these steps in the past and I pointed them out to you guys in the recording. You can view the cloud recording of the session [here](https://zoom.us/rec/share/UV3PAmIvdwBjXa5ic1isay6RSXHMNsjmYBAY8OX1mQ8t_NXpEBcpk4WNYDveyvfx.RpWhVNEiUWaLrLX1).

Also please find attached the notebook that I re-coded out and I will be pushing the notebook to the Instructor Resources as well.

[Our deployed LSTM that we developed in Class!!](https://huggingface.co/spaces/firobeid/shakespear-lstm)