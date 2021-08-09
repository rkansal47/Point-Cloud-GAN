import torch
import os

results_directory = '/graphganvol/Point-Cloud-GAN/results/'
save_directory = '/graphganvol/mnist_graph_gan/jets/models/pcgan/'
trainings = os.listdir(results_directory)

for training in trainings:
    full_path = results_directory + training + '/'
    models = os.listdir(full_path)
    filtered = [int(model[:-4].split('_')[-1]) for model in models if model[-4:] == '.pth']
    final_model = filtered.sort()[-1]
    G_inv = torch.load(f"{full_path}/G_inv_network_{final_model}.pth")
    G_pc = torch.load(f"{full_path}/G_network_{final_model}.pth")

    torch.save(G_inv.state_dict(), f"{full_path}/G_inv_final.pt")
    torch.save(G_pc.state_dict(), f"{full_path}/G_pc_final.pt")

    key = training[-1]
    if key == '1': key = 'g'
    torch.save(G_inv.state_dict(), f"{full_path}/pcgan_G_inv_{key}.pt")
    torch.save(G_pc.state_dict(), f"{full_path}/pcgan_G_pc_final_{key}.pt")
