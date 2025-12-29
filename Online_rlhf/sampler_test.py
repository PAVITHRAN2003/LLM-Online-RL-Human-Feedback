from sampler import generate_verified_traces

q, traces = generate_verified_traces("data/sample1.json")
print("Question:", q)
print("Number of verified traces:", len(traces))
for i, t in enumerate(traces):
    print(f"\nTrace {i+1}:\n{t[:200]}...")  # print first 200 chars
