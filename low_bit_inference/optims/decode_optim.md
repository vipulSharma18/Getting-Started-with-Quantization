Note on efficient decoding:
-----------------------------
HF generate by default uses a greedy decoding strategy.

1. We can do speculative decoding to optimize the inference if the need be.

2. Guided or structured text generation can also speed-up inference at the cost of
constraining the user's interaction with the model.