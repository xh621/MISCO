import os, time
import numpy
import numpy as np
import shutil
import random
import math
import torch

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo.run import run_ppo_sjr
from ppo.arguments import get_args

from evogym import sample_robot, hashable, BASELINE_ENV_NAMES
from evogym.utils import is_connected, has_actuator, get_full_connectivity

import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure

from vec2morph import vec_to_morph, morph_to_vec, operator, morph_to_organ # 这些函数是用来在VAE的0-1编码格式和形态的格式（即5*5）之间来回切换的

import pyro
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    JitTraceEnum_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    config_enumerate,
)
from pyro.optim import Adam

# 生成新的种群
def sample_population(prob_dist, pop_size=25, grid_size=5, material_types=5):
    new_population = np.zeros((pop_size, grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            new_population[:, i, j] = np.random.choice(
                material_types, size=pop_size, p=prob_dist[i, j]
            )
    return new_population#产生的就是morph

def initialize_population(pop_size=25, grid_size=5, material_types=5):
    return np.random.randint(0,material_types,(pop_size,grid_size,grid_size))
    
# 选择精英样本更新概率分布
def select_elite(population, fitness, elite_frac=0.6):
    elite_count = int(elite_frac * len(population))
    elite_indices = np.argsort(fitness)[-elite_count:]  # 适应度越大越好
    return population[elite_indices]

# 选择精英样本更新概率分布
def select_elite_from_pool(all_bodies, fitnesses, tau,operator_2,width,pop_size,generation):#传入的all_bodies, fitnesses都是全局资源池的数据
   # sample_size = 50 
    sample_size = min(int(len(fitnesses[0])*0.5), 50)
    # sample_adv = 30,当代数逐渐增加，应该保持优势id样本在指导vae的过程中占据至少70%的比例，
    sample_adv = min(int(len(fitnesses[0])*0.1), 35)
    sample_probs = {}
    
    for k in range(args.num_tasks):
        population = vec_to_morph(all_bodies[k],operator_2,width)
        sample_prob = np.exp(np.array(fitnesses[k])*tau) / np.exp(np.array(fitnesses[k])*tau).sum()
        sample_probs[k] = sample_prob

    for k in range(args.num_tasks):
        sample_advs = sorted(range(len(fitnesses[k])), key=lambda i : fitnesses[k][i], reverse=True)[:sample_adv]
        sample_ids = np.random.choice(range(pop_size*(generation+1)), sample_size-sample_adv, replace=True, p=sample_probs[k])
        sample_ids = np.concatenate((sample_advs,sample_ids))#优势形态加多样性
        if k==0:
            sample_bodies = population[sample_ids,:]
        else:
            sample_bodies = torch.cat((sample_bodies, population[sample_ids,:]), dim=0)
    return sample_bodies
                    
# 更新概率分布参数（极大似然估计）
def update_distribution(elite_samples, grid_size=5, material_types=5):#elite_samples也是机器人形态，是体素材料
    # 初始化多项分布参数
    print(elite_samples)
    prob_dist = np.zeros((grid_size, grid_size, material_types))
    
    # 统计精英样本中每种材料类型出现的频率
    for i in range(grid_size):
        for j in range(grid_size):
            for sample in elite_samples:
                #print(sample[i, j],type(sample[i, j]),'jj')
                prob_dist[i, j, int(sample[i, j])] += 1
                
    # 归一化以得到概率
    prob_dist = prob_dist / np.sum(prob_dist, axis=2, keepdims=True)
    return prob_dist

def run_vae(experiment_name, structure_shape, pop_size, max_evaluations, train_iters, num_cores, args, generations):
    
    # =======================================================================================
    # 1. Initialize the model
    # =======================================================================================
    operator_2 = operator(args.width) 
    # ========================================================================================
    # =======================================================================================
    
    # initialize a dictionary used for storing fitness scores 
    #  (to calculate sample probabilities in continuous natural selection)
    fitnesses = {}
    ids = {}
    for k in range(args.num_tasks):
        fitnesses[k] = []
        ids[k] = []
    
    random_prop = 0  # sample robots with VAE, rather than the built-in function in EvoGym
    
    for generation in range(generations):
        torch.cuda.empty_cache()
        
        ### STARTUP: MANAGE DIRECTORIES ###
        home_path = os.path.join(root_dir, "saved_data", experiment_name)
        start_gen = 0

        ### DEFINE TERMINATION CONDITION ###    
        tc = TerminationCondition(train_iters)
        is_continuing = False  

        try:
            os.makedirs(home_path)
        except:
            if generation == 0:
                print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
                print("Override? (y/n/c): ", end="")
                ans = input()
                if ans.lower() == "y":
                    shutil.rmtree(home_path)
                    print()
                elif ans.lower() == "c":
                    print("Enter gen to start training on (0-indexed): ", end="")
                    start_gen = int(input())
                    is_continuing = True
                    print()
                else:
                    return
            else:
                pass


        ### STORE META-DATA ##
        if not is_continuing:
            temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
            vae_loss_path_sup = os.path.join(root_dir, "saved_data", experiment_name, "vae_loss_sup.txt")
            vae_loss_path_unsup = os.path.join(root_dir, "saved_data", experiment_name, "vae_loss_unsup.txt")
            valid_num_path = os.path.join(root_dir, "saved_data", experiment_name, "valid_num.txt")
            
            try:
                os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))
            except:
                pass

            f = open(temp_path, "w")
            f.write(f'POP_SIZE: {pop_size}\n')
            f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
            f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
            f.write(f'TRAIN_ITERS: {train_iters}\n')
            f.write(f'CPU_SEED: {args.cpu_seed}\n')
            f.write(f'GPU_SEED: {args.seed}\n')
            f.write(f'sim_thr: {args.sim_thr}\n')
            f.write(f'T: {args.tau}\n')
            f.write(f'exploration: {args.explore_rate}\n')
            f.write(f'NUM_TASKS: {args.num_tasks}\n')
            f.write(f'NUM_ROBO_TYPES: {args.num_robo_types}\n')
            f.write(f'NUM_ORGAN_TYPES: {args.num_organs}\n')
            f.write(f'NUM_VOXEL_TYPES: {args.num_voxels}\n')

            f.close()

        else:
            pass

        ### UPDATE NUM SURVIORS ###		
        num_evaluations = pop_size * generation # number of evaluated robot designs
        survivor_rate = 1 # keep all robot designs in the morphology pool
        num_survivors = pop_size
        #explore = math.ceil(pop_size*args.explore_rate*(generations-generation-1)/(generations-1))
        
        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "structure")
        save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller")
        save_path_organ = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "organ")
        save_path_type = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation),"type")
        
        try:
            os.makedirs(save_path_structure)
            os.makedirs(save_path_controller)
            os.makedirs(save_path_organ)
            os.makedirs(save_path_type)
        except:
            pass
        
        structures = []
        robo_type_dist = {}
        nums_robots = {}
        nums_robots_valid = {}
        
        for task_id in range(args.num_tasks):
            nums_robots[task_id] = 0
            nums_robots_valid[task_id] = 0
        
        for task_id in range(args.num_tasks):
            population_structure_hashes = {}
            tasks = torch.zeros((pop_size, args.num_tasks))
            tasks[:,task_id] = 1
           
        # =======================================================================================
        #    2. generate morphologies using Muilti
        # =======================================================================================
            # 初始化种群
            if generation==0:
                flag='ran'
                robos_vec = initialize_population(pop_size,args.grid_size,args.material_types)      
                    ## keep record of the proportion of valid robots
            else:
                flag='mul'
                robos_vec = sample_population(prob_dist, pop_size,args.grid_size,args.material_types)
            robos = robos_vec#vec_to_morph(robos_vec, operator_2, args.width)

            for robo_idx in range(robos.shape[0]):
                temp_robo = robos[robo_idx,:,:]
                if is_connected(temp_robo) and has_actuator(temp_robo):
                    nums_robots_valid[task_id] += 1

            print(robos.shape[0])
            for robo_idx in range(robos.shape[0]):
                if len(population_structure_hashes) == pop_size:
                        break
                temp_robo = robos[robo_idx,:,:]

                if is_connected(temp_robo) and has_actuator(temp_robo) and hashable(temp_robo) not in population_structure_hashes: 
                    # exploration-exploitation rebalancing (except for the first generation)

                    temp_structure = (temp_robo, get_full_connectivity(np.array(temp_robo)))
                    structures.append(Structure(*temp_structure, str(len(population_structure_hashes))+flag, task_id=task_id))
                    population_structure_hashes[hashable(temp_structure[0])] = True


            if len(population_structure_hashes) != pop_size and generation == 0:
                # upper bound of attempts met, but not enough exploitation samples: directly generate with ran
                flag='ran'
                while len(population_structure_hashes) != pop_size:
                    robos_vec = initialize_population(pop_size,args.grid_size, args.material_types)
                    robos = robos_vec#vec_to_morph(robos_vec, operator_2, args.width)

                    for robo_idx in range(robos.shape[0]):
                        if len(population_structure_hashes) == pop_size:
                            break
                        temp_robo = robos[robo_idx,:,:]
                        if is_connected(temp_robo) and has_actuator(temp_robo) and hashable(temp_robo) not in population_structure_hashes: 
                            temp_structure = (temp_robo, get_full_connectivity(np.array(temp_robo)))
                            structures.append(Structure(*temp_structure, str(len(population_structure_hashes))+flag, task_id=task_id))
                            population_structure_hashes[hashable(temp_structure[0])] = True
                            
            if len(population_structure_hashes) != pop_size and generation > 0:
                flag='mul'
                # upper bound of attempts met, but not enough exploitation samples: directly generate with Multi
                while len(population_structure_hashes) != pop_size:
                    robos_vec = sample_population(prob_dist,pop_size,args.grid_size,args.material_types)
                    robos = robos_vec#vec_to_morph(robos_vec, operator_2, args.width)

                    for robo_idx in range(robos.shape[0]):
                        if len(population_structure_hashes) == pop_size:
                            break
                        temp_robo = robos[robo_idx,:,:]
                        if is_connected(temp_robo) and has_actuator(temp_robo) and hashable(temp_robo) not in population_structure_hashes: 
                            temp_structure = (temp_robo, get_full_connectivity(np.array(temp_robo)))
                            structures.append(Structure(*temp_structure, str(len(population_structure_hashes))+flag, task_id=task_id))
                            population_structure_hashes[hashable(temp_structure[0])] = True
        # =======================================================================================
        # =======================================================================================
        
        
        ### SAVE POPULATION DATA ###
        # save valid proportion
        for task_id in range(args.num_tasks):
            f = open(valid_num_path, "a")
            out = ""
            out += str(generation) + "\t\t" + str(task_id) + "\t\t" + str(nums_robots[task_id]) + "\t\t" + str(nums_robots_valid[task_id]) + "\t\t" + str(nums_robots_valid[task_id]+nums_robots[task_id]) + "\n"
            f.write(out)
            f.close()
        
        # save robot designs
        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, "task-{}_struct-{}".format(structures[i].task_id, structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

    # =======================================================================================
    # 3. robot evaluation and saving 
    # =======================================================================================
        #better parallel
        group = mp.Group()
        for structure in structures:
        # int(structure.label[:-3])%2+2
            ppo_args = (1, args, task_id_name_mapper[structure.task_id], structure, tc, (save_path_controller, structure.label))
            group.add_job(run_ppo_sjr, ppo_args, callback=structure.set_reward)
        group.run_jobs(num_cores) 

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()

        tasks_structures = []
        for task_id in range(args.num_tasks):
            task_structures = []
            for robo_idx in range(pop_size):
                task_structures.append(structures[task_id * pop_size + robo_idx]) 
            task_structures = sorted(task_structures, key=lambda task_structure: task_structure.fitness, reverse=True) 
            tasks_structures.append(task_structures)

        #SAVE RANKING TO FILE
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "output"+str(generation)+".txt")
        f = open(temp_path, "w")

        out = ""
        for task_structures in tasks_structures:
            for task_structure in task_structures:
                out += str(task_structure.task_id) + "\t\t" + str(task_structure.label) + "\t\t" + str(task_structure.fitness) + "\n"
        f.write(out)
        f.close()

        ### CHECK EARLY TERMINATION ###
        if num_evaluations >= max_evaluations:
            print(f'Trained exactly {num_evaluations} robots')
            return

        ### SAVE DESIGNS AND FITNESS INTO THE MORPHOLOGY POOL ###
        tasks_survivors = []
        for k, task_structures in enumerate(tasks_structures):
            task_survivors = task_structures[:num_survivors]
            tasks_survivors.append(task_survivors)
            fitnesses[k] += [kk.fitness for kk in task_survivors]
            ids[k] += [str(generation)+" "+str(kk.label)[:-3] for kk in task_survivors]
        
        survivor_bodies = []
        for task_survivors in tasks_survivors:
            for survivor in task_survivors:
                survivor_bodies.append(np.array(survivor.body))
        survivor_bodies = torch.tensor(np.array(survivor_bodies))
        
        survivor_tasks = [torch.randint(1,(1,num_survivors)) + task_id for task_id in range(args.num_tasks)]#high=1，这里是0+【0,1,2】
        survivor_tasks = torch.hstack(survivor_tasks) 
        survivor_tasks_vec = []
        for t in survivor_tasks.squeeze():
            survivor_task = [0]*args.num_tasks#任务的onehot编码
            survivor_task[t] = 1
            survivor_tasks_vec.append(survivor_task)
        survivor_tasks = torch.tensor(survivor_tasks_vec)
        
        
        if generation == 0:
            all_bodies = {}
            all_tasks = {}
            for k in range(args.num_tasks):
                all_bodies[k] = morph_to_vec(survivor_bodies)[(pop_size*k):(pop_size*(k+1)),:].float()
                all_tasks[k] = survivor_tasks[(pop_size*k):(pop_size*(k+1)),:]
        else:
            for k in range(args.num_tasks):
                all_bodies[k] = torch.cat((all_bodies[k], morph_to_vec(survivor_bodies)[(pop_size*k):(pop_size*(k+1)),:].float()), dim=0)
                all_tasks[k] = torch.cat((all_tasks[k], survivor_tasks[(pop_size*k):(pop_size*(k+1)),:]), dim=0)
        
    # =======================================================================================
    # =======================================================================================
        
        
    # =======================================================================================
    # 4. update parameters through maximize likehood-update_distribution
    # =======================================================================================
        for k in range(args.num_tasks):
            # 选择精英样本，从这一代的种群中取
            elite_samples = select_elite_from_pool(all_bodies,fitnesses,args.tau,operator_2,args.width,pop_size,generation)
            #elite_samples = select_elite(survivor_bodies[(pop_size*k):(pop_size*(k+1)),:].float(), fitnesses[k][-pop_size:], args.elite_frac)
            # 更新概率分布
            prob_dist = update_distribution(elite_samples, args.grid_size, args.material_types)
            print('prob_dist :',prob_dist )            
        
        print("Updating finished! ")
    # =======================================================================================
    # =======================================================================================

    
    
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    seed = 2 # 1,2#注意torch的seed在get_arg()里面，也就是ppo文件夹的argument_sjr里面还有个seed,下面接口调出来直接改了
    random.seed(seed)
    np.random.seed(seed)

    structure_shape=(5, 5)
    pop_size = 25#25
    
    max_evaluations=1e100
    train_iters =1000
    num_cores =25
    generations = 40
    env_tasks = ["Flipper-v0"]#Walker-v0;Pusher-v0;Climber-v0
    task_name_id_mapper = {}
    task_id_name_mapper = {}
    for env_task in env_tasks:
        task_name_id_mapper[env_task] = len(task_name_id_mapper) 
        task_id_name_mapper[task_name_id_mapper[env_task]] = env_task 

    args = get_args() 
    args.elite_frac=0.6
    args.grid_size=5
    args.material_types=5
    args.seed = 1
    args.cpu_seed = 2
    args.width = structure_shape[0]
    args.num_tasks = len(env_tasks)
    args.num_voxels = 5

    args.hidden_layers_g = [128, 128, 128]
    args.hidden_layers_p = [128 ,128, 128]
    args.heads = 2
   
    # number of updates of VAE in each generation linearly increases from low to up
    args.num_iters_vae_low = 50 # 50 , 100
    args.num_iters_vae_up = 250 # 250, 300
    
    args.num_processes = 4
    args.num_steps = 128
    args.vae_lr = 0.0001
    args.vae_betas = (0.95,0.999)
    
    # Two variants: MorphVAE-H and -L
    args.tau = 1.5 # 0.7 
    args.sim_thr = 0 # 0.75
    args.explore = 0
    args.explore_rate = 0 # 0.5
    
    args.no_cuda = True
    args.cuda = False#here
    
    experiment_name = "CE_adv1_flipper(seed:2,1: no MP)" + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    run_vae(experiment_name, structure_shape, pop_size, max_evaluations, train_iters, num_cores, 
        args=args, generations=generations)
