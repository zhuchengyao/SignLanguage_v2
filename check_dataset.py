import os, json, sys, textwrap

root = "./datasets/ASL_gloss/train"   # ← 如果数据不在这里，请改成你的 train 目录

samples = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
print("train 目录子文件夹数量:", len(samples))

for d in samples[:5]:                 # 只检查前 5 个
    folder = os.path.join(root, d)
    txt = os.path.exists(os.path.join(folder, "text.txt"))
    js  = os.path.exists(os.path.join(folder, "pose.json"))
    print(f"\n📂 {d}: text.txt={txt}  pose.json={js}")
    if not js:
        continue
    poses = json.load(open(os.path.join(folder, "pose.json"), "r", encoding="utf-8")).get("poses", [])
    if not poses:
        print("  pose.json 里没有 'poses' 字段或为空"); continue

    fr = poses[0]                 # 只看第一帧
    print("  body len:", len(fr.get("pose_keypoints_2d", [])))
    print("  right hand len:", len(fr.get("hand_right_keypoints_2d", [])))
    print("  left  hand len:", len(fr.get("hand_left_keypoints_2d", [])))

    print("  sample body data:", textwrap.shorten(str(fr.get('pose_keypoints_2d', [])[:6]), 60))
    break
