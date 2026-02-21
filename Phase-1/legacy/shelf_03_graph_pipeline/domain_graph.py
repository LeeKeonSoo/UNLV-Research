from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

# Test text
text = "Solve the quadratic equation: x^2 + 5x + 6 = 0"

# Option 1: Detailed
domains_v1 = [
    "algebra and equations",
    "geometry and spatial reasoning",
    "physics and mechanics"
]

# Option 2: Simple  
domains_v2 = [
    "algebra",
    "geometry", 
    "physics"
]

result_v1 = classifier(text, domains_v1)
result_v2 = classifier(text, domains_v2)

print("V1:", result_v1['labels'][0], result_v1['scores'][0])
print("V2:", result_v2['labels'][0], result_v2['scores'][0])