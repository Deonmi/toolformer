# main_sample.py

import pandas as pd
from sampling import sampling_phase_for_text

if __name__ == "__main__":
    x_text = (
        'Below you\'ll find a list of all posts that have been categorized as '
        '"Improving Consistency of Performance"'
    )

    I, calls_by_pos = sampling_phase_for_text(x_text)
    # calls_by_pos is a dict, key is position, value is a list of SampledCall

    print("candidate position I:", I)
    rows = []
    for pos, calls in calls_by_pos.items():
        for c in calls:
            print(f"pos={pos}, api={c.api_name}, input={c.api_input}")
            rows.append(
                {
                    "text": x_text,
                    "position": pos,
                    "api_name": c.api_name,
                    "api_input": c.api_input,
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(
            "toolformer_sampling_output.tsv",
            sep="\t",
            index=False,
            mode="w",
            header=True
        )
