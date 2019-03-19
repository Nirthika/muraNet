import random
import csv

####################################################################################################
# Image path with positive negative LABEL
# Example: MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png,1

# with open('./data/MURA-v1.1/valid_image_paths.csv','r') as csvinput:
#     with open('./data/MURA-v1.1/valid.csv', 'w') as csvoutput:
#         writer = csv.writer(csvoutput, lineterminator='\n')
#         reader = csv.reader(csvinput)
#         data = []
#         for row in reader:
#             if "positive" in row[0]:
#                 row.append(1)
#                 data.append(row)
#             else:
#                 row.append(0)
#                 data.append(row)
#         # To shuffle data
#         # random.shuffle(data)
#         writer.writerows(data)

####################################################################################################
# Image path with positive negative LABEL and additional column with 7 classes
# Example: MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png,SHOULDER,1

with open('./data/MURA-v1.1/valid_image_paths.csv','r') as csvinput:
    with open('./data/MURA-v1.1/valid_ILC.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)
        data = []
        for row in reader:
            if "SHOULDER" in row[0]:
                row.append("SHOULDER")
            elif "HUMERUS" in row[0]:
                row.append("HUMERUS")
            elif "FINGER" in row[0]:
                row.append("FINGER")
            elif "ELBOW" in row[0]:
                row.append("ELBOW")
            elif "WRIST" in row[0]:
                row.append("WRIST")
            elif "FOREARM" in row[0]:
                row.append("FOREARM")
            elif "HAND" in row[0]:
                row.append("HAND")

            if "positive" in row[0]:
                row.append(1)
                data.append(row)
            else:
                row.append(0)
                data.append(row)
        # To shuffle data
        # random.shuffle(data)
        writer.writerows(data)

####################################################################################################
# Image path with 14 LABEL indicating class and positive or negative
# Example: MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png,SHOULDER_P

# with open('./data/MURA-v1.1/valid_image_paths.csv','r') as csvinput:
#     with open('./data/MURA-v1.1/valid.csv', 'w') as csvoutput:
#         writer = csv.writer(csvoutput, lineterminator='\n')
#         reader = csv.reader(csvinput)
#         data = []
#         for row in reader:
#             if "SHOULDER" in row[0]:
#                 if "positive" in row[0]:
#                     row.append("SHOULDER_P")
#                     data.append(row)
#                 else:
#                     row.append("SHOULDER_N")
#                     data.append(row)
#             elif "HUMERUS" in row[0]:
#                 if "positive" in row[0]:
#                     row.append("HUMERUS_P")
#                     data.append(row)
#                 else:
#                     row.append("HUMERUS_N")
#                     data.append(row)
#             elif "FINGER" in row[0]:
#                 if "positive" in row[0]:
#                     row.append("FINGER_P")
#                     data.append(row)
#                 else:
#                     row.append("FINGER_N")
#                     data.append(row)
#             elif "ELBOW" in row[0]:
#                 if "positive" in row[0]:
#                     row.append("ELBOW_P")
#                     data.append(row)
#                 else:
#                     row.append("ELBOW_N")
#                     data.append(row)
#             elif "WRIST" in row[0]:
#                 if "positive" in row[0]:
#                     row.append("WRIST_P")
#                     data.append(row)
#                 else:
#                     row.append("WRIST_N")
#                     data.append(row)
#             elif "FOREARM" in row[0]:
#                 if "positive" in row[0]:
#                     row.append("FOREARM_P")
#                     data.append(row)
#                 else:
#                     row.append("FOREARM_N")
#                     data.append(row)
#             elif "HAND" in row[0]:
#                 if "positive" in row[0]:
#                     row.append("HAND_P")
#                     data.append(row)
#                 else:
#                     row.append("HAND_N")
#                     data.append(row)
#
#         # To shuffle data
#         # random.shuffle(data)
#         writer.writerows(data)
