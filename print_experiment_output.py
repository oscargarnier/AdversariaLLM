import json
import matplotlib.pyplot as plt

file = "outputs/2026-07-16/11-11-59/0/run.json"
with open(file, "r") as f:
    results = json.load(f)

def print_dic(dic):
    for key, value in dic.items():
        print(f"{key}: {value}")

def pretty_print_jailbreak(dic):
    print("PROMPT")
    print(dic['model_input'][0]['content'])
    print(f"OUTPUT")
    print(dic["model_completions"][0])

def plot_keyword(steps, keyword):
    values = [steps[i][keyword] for i in range(len(steps))]
    plt.plot(values)
    plt.xlabel("step")
    plt.ylabel(keyword)
    plt.show()

def plot_scores(steps, judge_name):
    values = [steps[i]["scores"][judge_name]["p_harmful"] for i in range(len(steps))]
    plt.plot(values)
    plt.xlabel("step")
    plt.ylabel("p_harmful")
    plt.show()

def get_best_score_index(steps, judge_name):
    values = [steps[i]["scores"][judge_name]["p_harmful"] for i in range(len(steps))]
    best_index = values.index(max(values))
    return best_index

runs = results["runs"]
run = runs[0]
steps = run["steps"]
index = get_best_score_index(steps, "strong_reject")

print(f"Best step index: {index}")
pretty_print_jailbreak(steps[index])