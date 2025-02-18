from datasets import load_dataset
from datasets import Dataset
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm

def save_mp4(video_bytes, path):
    with open(path, 'wb') as video_file:
        video_file.write(video_bytes)

def process_dataset_chunks(dataset, chunk_size=5, max_chunks=10, save_dir="~/temp/"):
    # create directory if not exists
    save_mp4_path = os.path.join(save_dir, "mp4s")
    os.makedirs(save_mp4_path, exist_ok=True)
    # chunk size in seconds
    datetimer = lambda x: datetime.strptime(x, "%H:%M:%S.%f").time()
    dataset_list = []
    counter = 0
    for example in tqdm(dataset, total=len(dataset)):
        # save mp4
        save_path_str = os.path.join(save_mp4_path, f"{counter}.mp4")
        save_mp4(example["mp4"], save_path_str)
        counter +=1
        batch_item = {"chunks": []}
        # combine activities
        all_activities = []
        example = filtered_dataset[0]
        for j in range(len(example["json"]["content_metadata"]["scenes"])):
            all_activities += example["json"]["content_metadata"]["scenes"][j]["activities"]
        # convert timestamp start and end to int tuple
        for activity in all_activities:
            start = datetimer(activity["timestamp"]["start_timestamp"])
            start = start.second + start.minute * 60 + start.hour * 3600
            end = datetimer(activity["timestamp"]["end_timestamp"])
            end = end.second + end.minute * 60 + end.hour * 360
            activity["timestamp"]["interval"] = (start, end)
        # chunk intervals 
        max_seconds = example["json"]["duration_seconds"]
        chunked_intervals = [(chunk_size*i, chunk_size*(i+1)) for i in range(max_chunks)]
        # sample activity that happened before the end of the chunk
        rolling_activities = []
        idx = 0
        for chunk in chunked_intervals:
            if(chunk[1] > max_seconds):
                break
            for j, activity in enumerate(all_activities):
                if activity["timestamp"]["interval"][0] < chunk[1] and j > idx:
                    rolling_activities.append(activity)
                    idx += 1
            if(len(rolling_activities) == 0):
                sample_idx = np.random.randint(0, len(all_activities))
            else:
                sample_idx = np.random.randint(0, len(rolling_activities))
            batch_item["chunks"].append({"curr_interval": chunk,
                                         "activity_interval": all_activities[sample_idx]["timestamp"]["interval"], 
                                         "activity": all_activities[sample_idx]["description"],
                                         "mp4_path": save_path_str})
        batch_item["mp4"] = example["mp4"]
        dataset_list.append(batch_item)

    return dataset_list


# #full dataset (600GB of data)
dataset = load_dataset("HuggingFaceFV/finevideo", split="train", num_proc=64)
filtered_dataset = dataset.filter(lambda x: x['json']["content_parent_category"] == "Sports", num_proc=64)
save_path = "/home/ubuntu/temp"
processed_dataset = process_dataset_chunks(filtered_dataset, chunk_size=5, 
                                           max_chunks=20, save_dir=save_path)
processed_dataset_hf = Dataset.from_list(processed_dataset)
processed_dataset_hf.save_to_disk(f"{save_path}/sports_dataset")