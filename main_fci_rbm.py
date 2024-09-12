# ==============================================================================================
# Full Configuration Interaction optimization with RBM
# Authors: Jorge I. Hernandez-Martinez
# Affiliations:
# CINVESTAV Guadalajara, Department of Electrical Engineering and Computer Science, Jalisco, 45017, Mexico
# 
# This code is part of the research conducted for the paper titled:
# "Configuration Interaction Guided Sampling with Interpretable Restricted Boltzmann Machine"
# 
# ==============================================================================================



import numpy as np
import clean_data_qp
import format_to_qp
import matplotlib.pyplot as plt
import bash_commands
import time
import os
import multiprocessing
import random



class RBM:
    #Initializes the class with the network parameters. Weights are randomly initialized, while the biases for the visible and hidden layers are set to zero.
    def __init__(self, num_visible, num_hidden,temp):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.weights = np.random.randn(num_visible, num_hidden)
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)
        self.Ne=0
        self.temp=temp

    # sigmoid activation function with clipping to avoid overflow
    def sigmoid(self, x):
        x=np.clip(x,-500,500) 
        return 1.0 / (1.0 + np.exp(-x))
    
    # function to sample the hidden units
    def sample_hidden(self, visible):
        np.random.seed(os.getpid()) # set the seed for each multiprocess
        hidden_activations = np.dot(visible, self.weights) + self.hidden_bias   # hidden_activations = a_i + sum_j(W_ij * v_j)
        hidden_probs = self.sigmoid(hidden_activations/self.temp)   # hidden_probs = sigmoid(hidden_activations)  
        hidden_states = hidden_probs > np.random.random(hidden_probs.shape) # sample the hidden units, if the probability is greater than a random number between 0 and 1, the hidden unit is activated

        return hidden_probs, hidden_states

    # function to sample the visible units
    def sample_visible(self, hidden):
        visible_activations = np.dot(hidden, self.weights.T) + self.visible_bias    # visible_activations = b_i + sum_j(W_ij * h_j)
        visible_probs = self.sigmoid(visible_activations/self.temp) # visible_probs = sigmoid(visible_activations)
        visible_states = visible_probs > np.random.random(visible_probs.shape)  # sample the visible units, if the probability is greater than a random number between 0 and 1, the visible unit is activated
        return visible_probs, visible_states


    def sample_visible_tower_sampling(self, hidden):
        '''
        Tower Sampling  to sample the visible units----
            Using this algorithm, we can sample the visible units, using the constrictions of: closed-shell systems, have the same number of alpha and beta electrons,
            and not having two electrons with the same spin in the same orbital.

            input:
                - hidden: hidden units
            output:
                - visible_probs: probabilities of the visible units
                - visible_states: visible units
        '''

        np.random.seed(os.getpid()) # set the seed for each multiprocess

       # Compute the probabilities of the visible units
        visible_activations = np.dot(hidden, self.weights.T) + self.visible_bias
        visible_probs = self.sigmoid(visible_activations/self.temp) 
 
        visible_states = np.zeros_like(visible_probs)   #initialize the visible units to zeros, this is the vector of visible units sampled using Tower Sampling

        # Towe Sampling for visible units in even positions (beta electrons)---------------------
        for i in range(self.Ne // 2):
            random_values_even = np.random.random(visible_states.shape[0])  # random values for each batch
            cumsum_probs_even = np.cumsum(visible_probs[:, ::2], axis=1)    # cumulative sum of probabilities for each batch
            p = random_values_even * cumsum_probs_even[:, -1]  # random values between 0 and the sum of all the probabilities in the even positions
            j = np.argmax(cumsum_probs_even >= p[:, None], axis=1) * 2  # find the position of the cumulative sum of probabilities that is greater than the random value
            visible_states[np.arange(visible_states.shape[0]), j] = 1   # activate the visible unit in even position in j position
            visible_probs[np.arange(visible_probs.shape[0]), j] = 0 # once the visible unit is activated, the probability is set to zero, to avoid to activate it again
        

        #Tower Sampling for visible units in odd positions (alpha electrons)---------------------     
        for i in range(self.Ne // 2):
            random_values_odd = np.random.random(visible_states.shape[0])
            cumsum_probs_odd = np.cumsum(visible_probs[:, 1::2], axis=1)
            p = random_values_odd * np.sum(visible_probs[:, 1::2], axis=1)
            j = np.argmax(cumsum_probs_odd >= p[:, None], axis=1) * 2 + 1
            visible_states[np.arange(visible_states.shape[0]), j] = 1
            visible_probs[np.arange(visible_probs.shape[0]), j] = 0
                    
        return visible_probs, visible_states
    
    
    def generation_sample_visible_jumping_probs(self, hidden,dets_train):
        '''
        Tower Sampling to sample the visible units during the generation of new determinants----

            Using this algorithm, we can sample the visible units, using the constrictions of: closed-shell systems, have the same number of alpha and beta electrons,
            and not having two electrons with the same spin in the same orbital. In this case, we use the algorithm of jumping probabilities to generate new determinants that differ by 1 or 2 substitutions from a
            random determinant in the training set.

            input:
                - hidden: hidden units
                - dets_train: training set of determinants
            output:
                - new_det: new determinant generated with physical constraints (single and double excitations and closed-shell systems)

        '''
       # compute the probabilities of the visible units
        visible_activations = np.dot(hidden, self.weights.T) + self.visible_bias
        visible_probs = self.sigmoid(visible_activations/self.temp) 

        #random number to choose between single or double excitation
        rand_excitation = np.array([random.random()])
        exitation=0
        if rand_excitation[0] <= 0.5: # singles excitation
            exitation=1
        else:   # doubles excitation
            exitation=2


        
        #take random determinant from the training set
        random_number_det=random.randint(0,len(dets_train)-1) #creo que funciona mejor en el paralelismo
        det=dets_train[random_number_det]
        
        #positions of the occupied and unoccupied orbitals
        occupied_orbitals=np.where(det==1)[0]
        unoccupied_orbitals=np.where(det==0)[0]

        #compute the conditional probabilities to move an electron from an occupied orbital to an unoccupied orbital
        jumps=np.zeros((len(occupied_orbitals),len(unoccupied_orbitals))) #initialize the jumps matrix probabilities to zero
        for j in range(len(occupied_orbitals)):
            for k in range(len(unoccupied_orbitals)):
                #probabilities of the jumps, (1 - prob of the occupied orbital being ocuped)*(prob of the unoccupied orbital being ocuped)
                jumps[j][k]=(1-visible_probs[0][occupied_orbitals[j]])*visible_probs[0][unoccupied_orbitals[k]]

        
        #sample the move of the electrons using the jumping probabilities
        new_det=det.copy()
        copy_original_det=det.copy()
        for i in range(exitation):
            c=0 #variable to acumulate the probabilities
            p=random.random()*np.sum(jumps) #random number between 0 and the sum of all the probabilities
            found_change=False
            for j in range(len(occupied_orbitals)):
                for k in range(len(unoccupied_orbitals)):
                    c+=jumps[j][k]
                    #print('cumsum:',c, 'jumps',jumps[j][k])
                    
                    #if the random number is less or equal to the acumulative sum of probabilities, then we change the determinant
                    # the restriction jumps[j][k]!=0 is to avoid to choose the same occupied and unoccupied orbital, that previously was changed to 0
                    #for example (j=7 y k= 30 cumsum=112.07108543752757) and  (j=7 y k =31 cumsum=112.07108543752757), the probabilities 
                    #are the same, it means that the occupied orbital 7 was changed to 0, and the unoccupied orbital 30 was changed to 0, so we cannot choose it again
                    if p<=c and jumps[j][k]!=0: 
                        #check if the occupied and unoccupied orbitals have the same spin (alpha electron in alpha orbital or beta electron in beta orbital)
                        if (occupied_orbitals[j]%2==0 and unoccupied_orbitals[k]%2==0) or (occupied_orbitals[j]%2!=0 and unoccupied_orbitals[k]%2!=0):
                            #change the determinant
                            new_det[occupied_orbitals[j]]=0
                            new_det[unoccupied_orbitals[k]]=1

                            #change the probabilities for this orbitals to 0, to avoid to choose them again
                            jumps[j]=0
                            jumps[:,k]=0
                            found_change=True
                            
                            break

                if found_change:
                    break
            if not found_change:
                #this occurs when the random number is one of the last probabilities, and happens that the probabilities are zero because the orbitals were changed
                #or we cannot change the determinant because the orbitals have diferent spins in the last probabilities.
                #In this case we choose the last probabilities that satisfy the condition of the spins of the orbitals and the probabilities are not zero due to previous changes

                #find the last probabilities that satisfy the comented conditions
                
                for j in range(len(occupied_orbitals)-1,-1,-1):
                    for k in range(len(unoccupied_orbitals)-1,-1,-1):
                        if jumps[j][k]!=0:
                            if (occupied_orbitals[j]%2==0 and unoccupied_orbitals[k]%2==0) or (occupied_orbitals[j]%2!=0 and unoccupied_orbitals[k]%2!=0):
                                #change the determinant
                                new_det[occupied_orbitals[j]]=0
                                new_det[unoccupied_orbitals[k]]=1
                                #change the probabilities for this orbitals to 0, to avoid to choose them again
                                jumps[j]=0
                                jumps[:,k]=0
                                found_change=True
                                break
                    if found_change:
                        break
                continue

        return new_det       


    def contrastive_divergence(self, visible, learning_rate=0.1, k=1):
        # Positive phase
        positive_hidden_probs, positive_hidden_states = self.sample_hidden(visible) # sampling the hidden units from the visible units
        positive_associations = np.dot(visible.T, positive_hidden_probs)        # calculating the associations between the visible and hidden units, i.e., <v_i * h_j>_data

        # Negative phase (k steps of Gibbs sampling)
        for _ in range(k):
            negative_visible_probs, negative_visible_states = self.sample_visible_tower_sampling(positive_hidden_states)   # sampling the visible units from the hidden units in Gibbs sampling
            negative_hidden_probs, negative_hidden_states = self.sample_hidden(negative_visible_states)    # sampling the hidden units from the visible units in Gibbs sampling

        negative_associations = np.dot(negative_visible_states.T, negative_hidden_probs)      # calculating the associations between the visible and hidden units, i.e., <v_i * h_j>_model

        # Update weights and biases
        self.weights += learning_rate * (positive_associations - negative_associations) 
        self.visible_bias += learning_rate * np.mean(visible - negative_visible_states, axis=0) 
        self.hidden_bias += learning_rate * np.mean(positive_hidden_probs - negative_hidden_probs, axis=0)

    def reconstruct(self, v):
        h = self.sigmoid(np.dot(v, self.weights) + self.hidden_bias)
        reconstructed_v = self.sigmoid(np.dot(h, self.weights.T) + self.visible_bias)
        return reconstructed_v



    def train(self, data, num_epochs=10, batch_size=10, learning_rate=0.1, k=1):
        num_examples = data.shape[0]
        self.Ne=np.count_nonzero(data[0] == 1)

        suffle_data=data.copy()
        for epoch in range(num_epochs):
            np.random.shuffle(suffle_data)  # shuffle the determinants in the training set

            for batch_start in range(0, num_examples, batch_size):
                batch_end = batch_start + batch_size
                batch_data = suffle_data[batch_start:batch_end] 
                self.contrastive_divergence(batch_data, learning_rate, k)

            #compute loss
            #loss = np.mean(np.square(data - self.reconstruct(data)))
            #print(f"Epoch {epoch+1}: Loss = {loss}") 
            #print(f"Epoch {epoch+1} completed",end="\r")


def rbm_initialization(ezfio_path,temp,prune):
    '''
        Initialize the rbm class

        parameters:
            - ezfio_path: path to the ezfio folder
            - temp: temperature
            - prune: threshold to prune the coefficients of the determinants
        
        return:
            - rbm: rbm class initialized
    '''
    bash_commands.unzip_dets_coefs(ezfio_path)
    x_train=get_and_clean_data(ezfio_path,prune)
    num_visible = x_train.shape[1] 
    num_hidden = x_train.shape[1] 
    rbm = RBM(num_visible, num_hidden,temp)
    bash_commands.zip_dets_coefs(ezfio_path)

    return rbm
            


def get_and_clean_data(ezfio_path,prune):
    
    '''
        Clean the qp files:
            - convert from decimal to binary with the same number of digits

        parameters:
            - ezfio_path: path to the ezfio folder
        
        return:
            - numpy array with the training dataset(determinants in  binary format) 
            - create the file with the deleted determinants to avoid repeating them in the next iteration
    '''

    qp_folder=ezfio_path+'/determinants/'
    x_train=clean_data_qp.clean(qp_folder+'psi_det', qp_folder+'psi_coef',prune)

    return x_train



def det_generation(rbm,x_train,dets_list,num_dets=2000):
    '''
        Generation of determinants with single and double excitations restrictions

        The algorithm is the following:
            - Generate a base determinant with all zeros except the first number of electrons positions
            - Generate random visible units
            - Sample the hidden units
            - Sample the visible units
            - Convert the visible units to integers
            - Check if the number of substitutions is 1 or 2
            - If the number of substitutions is 1 or 2, append the determinant to the list of determinants
            - Repeat until we have the number of determinants that fulfill the conditions of single and double substitutions

        
        parameters:
            - rbm: the rbm class
            - x_train: training dataset
            - num_dets: number of determinants to generate
        
        return:
            - determinantes: list of determinants that fulfill the conditions of single and double substitutions

    '''

    m=0 #Counter of determinants generated

    #convert training vectors of 1 and 0 to decimal
    train_dec=[]
    for i in range(len(x_train)):
        train_dec.append(int("".join(map(str, x_train[i][::-1])), 2))

    
    #set the seed for each multiprocess
    random.seed(os.getpid())

    #run until we have the number of determinants that fulfill the conditions of single and double substitutions
    while (m<=num_dets-1):
        visible_states = np.random.rand(1, rbm.num_visible) #generate random visible units

        #two gibbs sampling
        _, hidden_states = rbm.sample_hidden(visible_states)
        _, visible_states = rbm.sample_visible_tower_sampling(hidden_states)
        _, hidden_states = rbm.sample_hidden(visible_states)

        #sample the visible units using the algorithm of jumping probabilities to generate new determinants that differ by 1 or 2 substitutions from a
        #random determinant in the training set
        det=rbm.generation_sample_visible_jumping_probs(hidden_states,x_train)

        #verify if the new determinant is not in the training set 
        det_dec=int("".join(map(str, det[::-1])), 2)
        if det_dec not in train_dec:
            #determinantes_dec.append(det_dec)
            #determinantes.append(det)  #used for non parallel version

            #is to verify if other process is adding the same determinant, but at the end is necessary a double check to avoid repeated determinants
            #if det not in dets_list:
            dets_list.append(det)   #append the new determinant to the shared list of determinants in the subprocess
            m+=1
            if m%1000==0:
                print('number of determinants generated',m,end='\r')
            if m==num_dets:
                print('number of determinants generated',m)

    
    #it is not necessary to return the determinants, because we are using a shared list in the subprocess


def plot_ground_energy(ground_energy_list, ezfio_name,FCI_energy=None):
    '''
        scatter Plot of ground energy-
        this shows the ground energy of each iteration

        parameters:
            - ground_energy_list: list of ground energies

        return:
            - plot of ground energy
    '''
    x=np.arange(len(ground_energy_list),dtype=int)
    plt.plot(x[1::],ground_energy_list[1::],'-o')

    if FCI_energy!=None:
        plt.axhline(y=FCI_energy, color='r', linestyle='-')

    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Computed Ground Energy in each iteration for "+ezfio_name)
    plt.legend(['Computed Ground Energy','FCI Energy'])
    plt.savefig('graphs/energy_plot/'+ezfio_name+'_ground_energy.png')
    plt.close()
    #plt.show()




def repited_determinants(determinantes, train):
    '''
        Identify repeated determinants:

        Function to remove the repeated determinants in the same set and the determinants that are in the training set:
            - convert vectors of 1 and 0 to decimal, is more efficient to compare
            - convert training vectors of 1 and 0 to decimal
            - first lets remove the repeated determinants in the same set, because during generation of determinants, we can have repeated determinants
            - find the new determinants that are not in the training set
        
        parameters:
            - determinantes: list of determinants
            - train: training dataset
        
        return:
            - final_dets: list of determinants that are not repeated in the same set and are not in the training set
    
    '''
    
    #convert vectors of 1 and 0 to decimal, is more efficient to compare
    determinantes_dec=[]
    for i in range(len(determinantes)):
        determinantes_dec.append(int("".join(map(str, determinantes[i][:][:][::-1])), 2))
    
    #convert training vectors of 1 and 0 to decimal
    train_dec=[]
    for i in range(len(train)):
        train_dec.append(int("".join(map(str, train[i][::-1])), 2))


    #first lets remove the repeated determinants in the same set, because during generation of determinants, we can have repeated determinants
    determinantes_dec, unique_indices=np.unique(determinantes_dec,return_index=True)
    determinantes_unique = determinantes[unique_indices]
    print('Number of determinants removed because they are repeated:',len(determinantes)-len(determinantes_unique))

    #find the new determinants that are not in the training set
    mask = np.isin(determinantes_dec, train_dec).astype(int)
    print('Number of determinants removed because they are in the training set:',np.count_nonzero(mask==1))

    #---------------------------------------------------------------------------------------------------
    #validate if the generated dets are not in the previous removed dets--------------------------------
    #---------------------------------------------------------------------------------------------------

    #read the deleted determinants file
    f = open('deleted_dets.txt', 'r')
    lines = f.readlines()
    f.close()
    lines=[int(line.strip()) for line in lines] #convert to int

    #find the new determinants that are not in the deleted determinants file
    determinantes_flip=np.fliplr(determinantes_unique.astype(int))
    #binary to decimal for a faster comparison
    determinantes_flip_dec=[int("".join(map(str, row)), 2) for row in determinantes_flip]
    mask2 = np.isin(determinantes_flip_dec, lines).astype(int)

    final_dets=determinantes_unique[np.logical_and(mask==0,mask2==0)]   #determinantes que no estan en el training set ni en el deleted_dets file
    print('Final number of determinants to be added to the training set:',len(final_dets))
    return final_dets


def get_energy(ezfio_path,calculation='cisd'):
    '''
        Get the energy from the qp output file

        parameters:
            - ezfio_path: path to the ezfio folder
            - calculation: type of calculation (cisd, scf, diagonalization)
        
        return:
            - energy: ground energy
    '''
    output_file=ezfio_path+'/determinants/qp.out'
    with open(output_file) as f:
        lines = f.readlines()

        if calculation=='cisd':
            for i in range(len(lines)):
                if 'CISD Energies' in lines[i]:
                    energy=lines[i+1].split()[-1]
                    break
        elif calculation=='scf':
            for i in range(len(lines)):
                if 'SCF energy' in lines[i]:
                    energy=lines[i].split()[-1]
                    break
        elif calculation=='diagonalization':
            for i in range(len(lines)):
                if 'Energy of state' in lines[i]:
                    energy=lines[i].split()[-1]
                    break

    energy=float(energy)
    print('Computed Ground Energy:',energy)

    return energy


def plot_DetDistribution(determinantes, x_train,ezfio_name,iteration):
    dets_to_plot=[]
    for i in range(len(determinantes)):
        dets_to_plot.append(np.reshape(determinantes[i],determinantes.shape[1]))

    # Compute the frequency of ones (occuped MO) in each position for all the determinants
    frecuencias_1 = [0] * len(dets_to_plot[0])  # Initialize with zeros for the distribution of the generated determinants
    frecuencias_2 = [0] * len(x_train[0])  # Initialize with zeros for the distribution of the training determinants

    # Sum the bits in each position for all the determinants
    for vector in dets_to_plot:
        for i, bit in enumerate(vector):
            frecuencias_1[i] += bit    

    for vector in x_train:
        for i, bit in enumerate(vector):
            frecuencias_2[i] += bit

    
    #figure with 300 dpi
    plt.figure(dpi=300)
    
    plt.figure(figsize=(18, 6))
    posiciones = np.arange(len(frecuencias_1))

    plt.bar(posiciones, frecuencias_1, width=0.6, label='Generated Determinants', color='b', alpha=0.7)
    plt.bar(posiciones + 0.6, frecuencias_2, width=0.6, label='Training Determinants', color='g', alpha=0.7)

    plt.xlabel("Molecular Orbitals")
    plt.ylabel("Occupancy frequency")
    plt.title('occupancy frequency in each MO in iteration '+str(iteration)+' of '+ezfio_name)
    plt.xticks(posiciones + 0.3, range(len(frecuencias_1)))
    plt.legend()
    #plt.show()
    plt.savefig('graphs/det_distribution/det_distribution_'+ezfio_name+'_iteration_'+str(iteration)+'.png')
    plt.close()

def plot_DetDistribution2(determinantes, x_train,ezfio_name,iteration,ezfio_path):
    '''
        Plot the distribution of the determinants but weighted using the coefficient of the determinants to weight the distribution
        -note: this was used to compare the distribution weighted and not weighted of the determinants
    '''

    #unzip the files psi_coef and psi_det in the ezfio folder
    bash_commands.unzip_dets_coefs(ezfio_path)

    #open file to read the coefficients (omit the first two lines)
    f = open(ezfio_path+'/determinants/psi_coef', 'r')
    lines = f.readlines()
    f.close()
    lines=lines[2:] #omit the first two lines

    #convert to float
    coeficientes=[float(line.strip()) for line in lines]

    #zip again the files psi_coef and psi_det in the ezfio folder
    bash_commands.zip_dets_coefs(ezfio_path)

    dets_to_plot=[]
    for i in range(len(determinantes)):
        dets_to_plot.append(np.reshape(determinantes[i],determinantes.shape[1]))
    
    
    # Compute the frequency of ones (occuped MO) in each position for all the determinants
    frecuencias_1 = [0] * len(dets_to_plot[0])  # Initialize with zeros for the distribution of the generated determinants
    frecuencias_2 = [0] * len(x_train[0])  # Initialize with zeros for the distribution of the training determinants

    # Sum the bits in each position for all the determinants
    for i in range(len(dets_to_plot)):
        for j in range(len(dets_to_plot[i])):
            if dets_to_plot[i][j]==1:
                frecuencias_1[j] += coeficientes[i]
    
    for i in range(len(x_train)):
        for j in range(len(x_train[i])):
            if x_train[i][j]==1:
                frecuencias_2[j] += 1
    

    posiciones = np.arange(len(frecuencias_1))
    plt.figure(figsize=(18, 6))
    posiciones = np.arange(len(frecuencias_1))

    #figure with 300 dpi
    plt.figure(dpi=300)

    #plot each bar as a subplot
    plt.bar(posiciones, frecuencias_1, width=0.6, label='Generated Determinants', color='b', alpha=0.7)
    plt.bar(posiciones + 0.6, frecuencias_2, width=0.6, label='Training Determinants', color='g', alpha=0.7)

    plt.xlabel("Molecular Orbitals")
    plt.ylabel("Occupancy frequency")
    plt.title('occupancy frequency in each MO in iteration '+str(iteration)+' of '+ezfio_name)
    plt.xticks(posiciones + 0.3, range(len(frecuencias_1)))
    plt.legend()
    #plt.show()

    plt.savefig('graphs/weighted_det_distribution/w_det_distribution'+ezfio_name+'_iteration_'+str(iteration)+'.png')
    plt.close()

    

def plot_time_per_dets(number_of_det_list,ezfio_name,times_per_iteration_list):
    '''
        Plot of time per number of determinants-
        this shows the time per iteration and the number of determinants generated in each iteration

        parameters:
            - number_of_det_list: list of number of determinants
            - ezfio_name: name of the ezfio folder (ex: h2o_631g.ezfio)
            - times_per_iteration_list: list of time per iteration

        return:
            - plot of time per iteration
    '''
    #figure with 300 dpi
    plt.figure(dpi=300)
    plt.plot(number_of_det_list,times_per_iteration_list,'-o',color='b')
    plt.xlabel("Number of determinants")
    plt.ylabel("Time (s)")
    plt.title("Computational Time per determinants number for "+ezfio_name)
    plt.savefig('graphs/time_per_dets/'+ezfio_name+'_time_per_dets.png')
    plt.close()
    #plt.show()



def main(working_directory,ezfio_path,qpsh_path,iterations=2,num_epochs=1, batch_size=10, learning_rate=0.01, k=1,times_num_dets_gen=2,prune=1e-12,tol=1e-5,temp=1,FCI_energy=None, times_max_diag_time=2):
    
    os.chdir(working_directory)
    print('Current working directory:',os.getcwd())
    
    ground_energy_list=[]
    number_of_det_list=[]
    times_per_iteration_list=[]
    print('The selected ezfio_path is :',ezfio_path)
    ezfio_name=ezfio_path.split('/')[-1]

    #delete the deleted_dets.txt file if exists, this file is used to store the determinants that are removed from the training set
    if os.path.exists('deleted_dets.txt'):
        os.remove('deleted_dets.txt')

    num_nucleos = multiprocessing.cpu_count()
    print("Number of cores available:", num_nucleos)

    for iteration in range (iterations):
        print('**********************************************************')
        print('************ iteration:',iteration,'***************************')
        print('**********************************************************')

        #compute scf and cisd in the first iteration
        if iteration==0:
            bash_commands.reset_ezfio(qpsh_path,ezfio_path)
            bash_commands.scf(qpsh_path,ezfio_path)
            ground_energy_list.append(get_energy(ezfio_path,calculation='scf'))
            bash_commands.cisd(qpsh_path,ezfio_path)
            ground_energy_list.append(get_energy(ezfio_path,calculation='cisd'))

            #initialize the rbm here to not reset the weights in each iteration
            rbm=rbm_initialization(ezfio_path,temp,prune)

        
        #if is not the first iteration, use the rbm to generate new determinants and then diagonalize 
        else:
            init_time=time.time()
            bash_commands.unzip_dets_coefs(ezfio_path) #unzip the files psi_coef and psi_det in the determinants of the ezfio folder

            #get the training dataset and the number of electrons in the molecule (in binary format) and 
            # the file with the deleted determinants to avoid repeating them in the next iteration
            x_train=get_and_clean_data(ezfio_path,prune)
            #initialize the rbm here to reset the weights in each iteration (you need to comment the previous initialization)----------
            #rbm=rbm_initialization(ezfio_path,temp,prune)


            rbm.train(x_train, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, k=k)  #train the rbm
            num_dets_gen=times_num_dets_gen*len(x_train)  #number of determinants to generate
            #num_dets_gen=40000   #just for testing

            #generate the new determinants with parallelization using multithreading-------------------------------------
            manager = multiprocessing.Manager()
            dets_list = manager.list() # shared list of determinants to append the new determinants generated in the subprocesses
            processes = []
            for i in range(num_nucleos):
                num_dets_per_process=num_dets_gen//num_nucleos
                p = multiprocessing.Process(target=det_generation, args=(rbm,x_train,dets_list,num_dets_per_process))
                processes.append(p)
                p.start()

            # Wait for all processes to finish
            for p in processes:
                p.join()
            
            # convert the shared list of determinants to a list
            determinantes = list(dets_list)
            print('number of determinants generated:',len(determinantes))

            #Validate if the new determinants, are not in the training set
            determinantes_2 = np.array(determinantes).astype(int).reshape(len(determinantes),rbm.num_visible)
            determinantes_3 = repited_determinants(determinantes_2, x_train.astype(int))
            

            #join the new determinants with the training set
            new_dets=np.concatenate((x_train,determinantes_3),axis=0)

            #convert the new determinants to qp format and add them to the qp files
            determinantes_3_toqp=np.fliplr(determinantes_3.astype(int))  #format_to_qp function needs the determinants in reverse order ex from 110000 to 000011 to a correct conversion to decimal               
            format_to_qp.formatting2qp(determinantes_3_toqp,ezfio_path)

            number_of_det_list.append(len(new_dets))

            #run the diagonalization
            bash_commands.zip_dets_coefs(ezfio_path)
            bash_commands.write_det_num(ezfio_path)

            #we can modify the threshold of the davidson algorithm depending on the number of determinants

            if len(new_dets)<20000:
                bash_commands.modify_threshold_davidson(ezfio_path,'1e-8')
            elif len(new_dets)>20000 and len(new_dets)<50000:
                bash_commands.modify_threshold_davidson(ezfio_path,'1e-8')
            elif len(new_dets)>50000:
                bash_commands.modify_threshold_davidson(ezfio_path,'1e-8')
            
            #2 times greater than previous time
            max_diag_time= times_max_diag_time*times_per_iteration_list[-1] if iteration>1 else 1000
            print('max_diag_time:',max_diag_time)
            process_completed=bash_commands.diagonalization(qpsh_path,ezfio_path,max_diag_time)

            #if the diagonalization process is completed successfully continue with the next steps
            if process_completed or iteration==1:
                
                #get the energy from the qp output file
                ground_energy_list.append(get_energy(ezfio_path,calculation='diagonalization'))

                end_time=time.time()

                plot_DetDistribution(determinantes_3, x_train,ezfio_name,iteration)
                plot_DetDistribution2(determinantes_3, x_train,ezfio_name,iteration,ezfio_path)

                times_per_iteration_list.append(end_time-init_time)


                #check if the energy is converged
                if iteration>1:
                    if abs(ground_energy_list[iteration +1]-ground_energy_list[iteration])<tol:
                        print('Energy converged at iteration',iteration,'to a value of',ground_energy_list[iteration+1])
                        break

                    # If the last energy is > 2% or 1% higher than the previous energy, stop the iterations. This observation comes from experiments,
                    # since when the energy change is too large, it indicates that the diagonalization did not converge properly and the algorithm
                    # has a high probability of not converging in the next iteration.
                    if abs(ground_energy_list[iteration +1] - ground_energy_list[iteration]) > 1e-1:
                        print('the Algorith has to stop because the diagonalization did not converge properly')
                        print('Energy converged at iteration',iteration,'to a value of',ground_energy_list[iteration])
                        #remove the last calculations
                        ground_energy_list.pop()
                        number_of_det_list.pop()
                        times_per_iteration_list.pop()

                        break
                    
                    '''
                    #if is 1e-4 of the FCI energy, then stop the iterations
                    if abs(ground_energy_list[iteration+1]-FCI_energy)<1e-4:
                        print('Energy converged at iteration',iteration,'to a value of',ground_energy_list[iteration+1])
                        break

                    if ground_energy_list[iteration+1]<FCI_energy:
                        print('Energy converged at iteration',iteration,'to a value of',ground_energy_list[iteration+1])
                        break
                    
                    #if the energy increases, then stop. Some experiments show that when the number of dets increase but the energy increases, the algorithm in the 
                    #next diagonalization have the probability to not converge, so is better to stop the iterations, this is a heuristic and is not always true
                    #but this occurs in some cases near the convergence
                    #maybe i can add a threshold, if the energy increases more than 1e-3, then stop for example
                    if ground_energy_list[iteration +1]>ground_energy_list[iteration]:  
                        print('Energy converged at iteration',iteration,'to a value of',ground_energy_list[iteration])
                        break
                    '''
            else:
                print('Diagonalization process not completed or not converged')
                #remove the last number of determinants because not exist time for the diagonalization process
                number_of_det_list.pop()
                break

            
    print('number of determinants in each iteration:',number_of_det_list)
    print('time per iteration:',times_per_iteration_list)

    plot_ground_energy(ground_energy_list,ezfio_name, FCI_energy)
    plot_time_per_dets(number_of_det_list,ezfio_name,times_per_iteration_list)
    


    #save ground_energy_list, number_of_det_list and times_per_iteration_list to a file
    with open('ouput_files/'+ezfio_name+'_ground_energy_list.txt', 'w') as f:
        for item in ground_energy_list:
            f.write("%s\n" % item)
    with open('ouput_files/'+ezfio_name+'_number_of_det_list.txt', 'w') as f:
        for item in number_of_det_list:
            f.write("%s\n" % item)
    with open('ouput_files/'+ezfio_name+'_times_per_iteration_list.txt', 'w') as f:
        for item in times_per_iteration_list:
            f.write("%s\n" % item)



    return ground_energy_list,number_of_det_list,times_per_iteration_list

    


    
if __name__=='__main__':
    Initial_time = time.time()

    ##-----------------------------------examples to test the code-----------------------------------
    ##------------------------------------------------------------------------------------------------

    #set woking directory
    working_directory='/home/ivan/Descargas/Python_Codes_DFT/paper_code_implementation/FCI_RBM'
    




    #path to the ezfio folder for the molecule------------------------
    #ezfio_path='/home/ivan/Descargas/solving_fci/to_diagonalize.ezfio' #c2 ccpvdz
    ezfio_path='/home/ivan/Descargas/QP_examples/h2o/h2o_631g.ezfio'   #h2o 6-31g
    #ezfio_path='/home/ivan/Descargas/QP_examples/h2o/h2o_ccpvdz.ezfio'   #h2o ccpvdz

    #path to the Quantum Package qpsh---------------------------------
    qpsh_path='/home/ivan/Descargas/qp2/bin/qpsh'
    ezfio_name=ezfio_path.split('/')[-1]

    #primeras pruebas con times det num 20, max iter 10, aprox davidson 1e-10,1e-6, y 1e-8, prune 1e-8
    max_iterations=3; num_epochs=3; batch_size=10; learning_rate=0.003; k=10;times_num_dets_gen=15;prune=1e-10;tol=1e-5;temp=1; times_max_diag_time=10

    #exanple of FCI energy for the molecule to compare the convergence
    #FCI_energy=-76.12237    #h2o 6-31g
    FCI_energy=-75.72984    #c2 ccpvdz
    #FCI_energy=-76.24195    #h2o ccpvdz
    

    ground_energy_list,number_of_det_list,times_per_iteration_list=main(working_directory,ezfio_path,qpsh_path,max_iterations,num_epochs, batch_size, learning_rate, 
                                                                        k, times_num_dets_gen,prune,tol,temp,FCI_energy,times_max_diag_time)
    print('Final Ground Energy List:',ground_energy_list)

    Final_time = time.time()
    print('Total execution time:',Final_time-Initial_time)


    