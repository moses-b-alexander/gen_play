
# from * import *


# TODO hyperparameter tuning











        # print(f"Before/After Step {step+1}:")
        # print(torch.cuda.memory_allocated() / 1e6, "MB allocated")
        # print(torch.cuda.memory_reserved() / 1e6, "MB reserved")
        # all_params = list(pf.parameters())
        # param_mem = sum(p.numel() * p.element_size() for p in all_params)
        # grad_mem = sum(p.numel() * p.element_size()
        #     for p in all_params if p.grad is not None)
        # print(f"Params: {param_mem/1e6:.2f} MB. Grads: {grad_mem/1e6:.2f} MB")
        # print()


        # total_norm = 0.0
        # for p in pf.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # final_norm = total_norm ** 0.5
        # print("Total grad norm:", np.round(final_norm, 3))

        # for name, p in pf.named_parameters():
        #     if p.grad is not None:
        #         gd = p.grad.data.norm().item()
        #         wt = p.data.norm().item()
        #         rt = ((gd) / (wt + epsilon))
        #         print(" -- ", name, " || ", np.round(gd, 2), " || ", np.round(np.log10(gd), 0), " || ", np.round(rt, 1), " .. ")
        # print()
        # print()
        # print()
        # print()


