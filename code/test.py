import pandas as pd
import datamanager.saveutil as saveutil

# test file to isolate small components of the code or data for visualization or debugging
values = saveutil.load_obj('../data/output/final3/train_features_gan_augmented_nosil.pkl')
df = pd.DataFrame(values)
df.to_csv('../data/output/final3/final3_train_features_gan_augmented_nosil.csv', index=False)