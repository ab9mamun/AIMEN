import numpy as np
def calculate_sparsity(real_example, cf):
  n = len(real_example)
  num_changes = 0
  for i in range(n):
    if real_example[i] != cf[i]:
      num_changes+=1
  #print(num_changes)
  return num_changes


def calculate_distance(real_example, cf, minval, maxval):
  real_example = (real_example - minval)/(maxval - minval)
  cf = (cf - minval)/(maxval - minval)
  #print(real_example - cf)
  return np.linalg.norm(real_example - cf, ord=2)

def find_metrics(df, minval, maxval, ens):
  n = df.shape[0]//2
  all_sparsities = []
  all_dists = []

  cf_indices = [i for i in range(1, n*2, 2)]

  cfs = df.values[:, :-1].astype(int)[cf_indices]
  pred_pairs = ens.predict(cfs)
  print(pred_pairs)
  preds_plain = [round(pair[1]) for pair in pred_pairs]
  print(preds_plain)
  acc = (n-np.count_nonzero(preds_plain))/len(preds_plain)

  for i in range(n):
    real_example = df.iloc[i*2].to_numpy()[:-1] #excluding the label
    cf = df.iloc[i*2+1].to_numpy()[:-1]

    sparsity = calculate_sparsity(real_example, cf)
    distance = calculate_distance(real_example, cf, minval, maxval)

    all_sparsities.append(sparsity)
    all_dists.append(distance)

  all_sparsities = np.array(all_sparsities)
  all_dists = np.array(all_dists)
  avg_spars = np.mean(all_sparsities)
  std_spars = np.std(all_sparsities)

  avg_dist = np.mean(all_dists)
  std_dist = np.std(all_dists)
  txt = f"Accuracy & Distance & sparsity\n${acc:.3f}$ & ${avg_dist:.3f} \\pm {std_dist:.3f}$ & ${avg_spars:.3f} \\pm {std_spars:.3f}$"
  print(txt)
  return txt