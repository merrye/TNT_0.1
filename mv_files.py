import os, shutil
from os.path import join as pjoin

root_dir = pjoin(os.getcwd(), "dataset/interm_data")

# input_path = pjoin(root_dir, "val_intermediate_8/raw")
# # print(input_path)

# for j in range(5):
#     input_path = pjoin(root_dir, "train_intermediate_%d/raw" % (j + 5))
#     print(input_path)
#     n = len(os.listdir(input_path))
#     for i, file in enumerate(os.listdir(input_path)):
#         fpath, fname = os.path.split(file)
#         dstfile = pjoin(input_path, fname)
#         srcfile = pjoin(root_dir, "train_intermediate_%d/raw" % (j), fname)
#         shutil.move(dstfile, srcfile)

# for i, file in enumerate(os.listdir(input_path)[:26]):
#     fpath, fname = os.path.split(file)
#     dstfile = pjoin(input_path, fname)
#     srcfile = pjoin(root_dir, "val_intermediate_%d/raw    " % 9, fname)
#     shutil.move(dstfile, srcfile)

# for j in range(5):
#     input_path = pjoin(root_dir, "train_intermediate_%d/raw" % (j))
#     print(input_path)
#     n = len(os.listdir(input_path))
#     for i, file in enumerate(os.listdir(input_path)[:(n//2)]):
#         fpath, fname = os.path.split(file)
#         dstfile = pjoin(input_path, fname)
#         srcfile = pjoin(root_dir, "train_intermediate_%d/raw" % (j + 5), fname)
#         shutil.move(dstfile, srcfile)



sum = 0
print("train: ")
for i in range(10):
    p_dir = pjoin(root_dir, "train_intermediate_%d" % i)
    if not os.path.exists(p_dir): os.mkdir(p_dir)
    p = pjoin(p_dir,"raw")
    if not os.path.exists(p): os.mkdir(p)
    sum += len(os.listdir(p))
    print(i, len(os.listdir(p)))
print(sum)
# print("val: ")
# for i in range(10):
#     p_dir = pjoin(root_dir, "val_intermediate_%d" % i)
#     if not os.path.exists(p_dir): os.mkdir(p_dir)
#     p = pjoin(p_dir,"raw")
#     if not os.path.exists(p): os.mkdir(p)
#     sum += len(os.listdir(p))
#     print(i, len(os.listdir(p)))
# print(sum)

# # print(len(os.listdir(pjoin(root_dir, "val_intermediate_%d/raw"))))
# for i in range(4):
#     print(len(os.listdir(pjoin(root_dir, "val_intermediate_%d/raw" % i))))

# print(len(os.listdir(pjoin(root_dir, "test_intermediate/raw"))))
# print(len(os.listdir(pjoin(root_dir, "val_intermediate/raw"))))

# input_path = pjoin(root_dir, "val_intermediate_2/raw")
# print(input_path)

# for i, file in enumerate(os.listdir(input_path)):
#     # pos = i // 20000
#     fpath, fname = os.path.split(file)
#     dstfile = pjoin(input_path, fname)
#     srcfile = pjoin(root_dir, "val_intermediate_%d/raw" % 3, fname)
#     shutil.move(dstfile, srcfile)
#     if i == 1: break


# for i in range(4):
#     print(len(os.listdir(pjoin(root_dir, "train_intermediate_%d/raw" % i))))

# print()

# for i in range(4):
#     print(len(os.listdir(pjoin(root_dir, "val_intermediate_%d/raw" % i))))