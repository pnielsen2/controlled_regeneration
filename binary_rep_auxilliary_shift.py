import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from datetime import datetime
import json
import logging
from scipy.stats import ortho_group

# define dataset
shift_target = torch.randint(0, 2, (535040, 1), dtype=torch.float32)

def measure_performance(network,shift_target,contraction,shift):
    preds = network.predict(contraction,shift)
    print("dataset size",len(preds))
    correct = 0
    total = 0
    for i in range(len(preds)):
        total+=1
        if preds[i]==shift_target[i]:
            correct+=1
    print("acc: ",correct/total*100,"%")
    

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quine_network.log'),
        logging.StreamHandler()
    ]
)

class OrthogonalLayer(nn.Module):
    """Fixed orthogonal layer that remains unchanged during training."""
    
    def __init__(self, size=20):
        super(OrthogonalLayer, self).__init__()
        # Generate a fixed orthogonal matrix
        orthogonal_matrix = torch.tensor(ortho_group.rvs(size), dtype=torch.float32)
        self.weight = nn.Parameter(orthogonal_matrix, requires_grad=False)
    
    def forward(self, x):
        return torch.mm(x, self.weight.t())

class QuineNetwork(nn.Module):
    """Neural network designed to self-replicate through controlled regeneration."""

    def __init__(self, 
                 layer_sizes=[20, 512, 512, 512, 1],
                 activation_function='hardtanh_plus_x',
                 initial_weights=None,
                 weight_init_method='orthogonal',
                 scale_factor=0.01,
                 use_orthogonal_layer=False):
        """
        Initialize the QuineNetwork.

        Args:
            layer_sizes: List of integers defining network architecture
            activation_function: Activation function to use
            initial_weights: Optional pre-initialized weights
            weight_init_method: Method for weight initialization
            scale_factor: Scaling factor for initial weights
            use_orthogonal_layer: Whether to include fixed orthogonal input layer
        """
        super(QuineNetwork, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.weight_init_method = weight_init_method
        self.scale_factor = scale_factor
        self.use_orthogonal_layer = use_orthogonal_layer
        
        # Add fixed orthogonal layer if specified
        if use_orthogonal_layer:
            self.orthogonal_layer = OrthogonalLayer(layer_sizes[0])
        
        # Create main network layers without biases
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False) 
            for i in range(len(layer_sizes)-1)
        ])
        
        self._initialize_weights(initial_weights)
        
        # Create binary representation dataset
        self.X = self._create_binary_dataset()
        
        # Initialize history tracking
        self.history = {
            'losses': [],
            'weight_stats': [],
            'convergence_rates': [],
            'weight_distributions': []
        }
        
    def _initialize_weights(self, initial_weights):
        """Initialize network weights using specified method."""
        for idx, layer in enumerate(self.layers):
            if initial_weights is not None and idx < len(initial_weights):
                layer.weight = nn.Parameter(initial_weights[idx])
            else:
                if self.weight_init_method == 'orthogonal':
                    nn.init.orthogonal_(layer.weight)
                elif self.weight_init_method == 'normal':
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                elif self.weight_init_method == 'uniform':
                    nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
                elif self.weight_init_method == 'xavier':
                    nn.init.xavier_uniform_(layer.weight)
                else:
                    raise ValueError(f"Unknown initialization method: {self.weight_init_method}")
                # Apply scale factor
                layer.weight.data *= self.scale_factor
    
    def _create_binary_dataset(self):
        """Create binary encoding for all weights in the network."""
        bits_weight_type = 2
        bits_input_neuron = 9
        bits_output_neuron = 9

        total_bits = bits_weight_type + bits_input_neuron + bits_output_neuron
        dataset = []

        for layer_number in range(len(self.layer_sizes)-1):
            for output_index in range(self.layer_sizes[layer_number+1]):
                for input_index in range(self.layer_sizes[layer_number]):
                    weight_type_bits = self._int_to_bits(layer_number, bits_weight_type)
                    input_neuron_bits = self._int_to_bits(input_index, bits_input_neuron)
                    output_neuron_bits = self._int_to_bits(output_index, bits_output_neuron)

                    datapoint = weight_type_bits + input_neuron_bits + output_neuron_bits
                    dataset.append(datapoint)

        return torch.FloatTensor(dataset)

    def _int_to_bits(self, n, bits):
        """Convert integer to binary representation."""
        return [((n >> i) & 1) * 2 - 1 for i in reversed(range(bits))]

    def forward(self, x):
        """Forward pass through the network."""
        if self.use_orthogonal_layer:
            x = self.orthogonal_layer(x)
            
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.apply_activation(x)
                
        x = self.layers[-1](x)
        return x

    def apply_activation(self, x):
        """Apply the specified activation function."""
        if self.activation_function == 'hardtanh_plus_x':
            return torch.nn.functional.hardtanh(x) + x
        elif self.activation_function == 'hardtanh':
            return torch.nn.functional.hardtanh(x)
        elif self.activation_function == 'relu_plus_x':
            return torch.relu(x) + x
        elif self.activation_function == 'tanh':
            return torch.tanh(x)
        elif self.activation_function == 'sin':
            return torch.sin(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_function}")

    def regenerate(self, contraction, shift):
        """Perform one step of self-replication."""
        out = self(self.X)
        out = out / contraction + shift_target*shift # 

        layer_weights = []
        idx = 0
        for i in range(len(self.layer_sizes)-1):
            num_weights = self.layer_sizes[i+1] * self.layer_sizes[i]
            weight = out[idx:idx+num_weights].view(self.layer_sizes[i+1], self.layer_sizes[i]).detach()
            layer_weights.append(weight)
            idx += num_weights

        for i in range(len(self.layers)):
            self.layers[i].weight = nn.Parameter(layer_weights[i])

        return out

    def predict(self,contraction,shift):
        out = self(self.X)/contraction
        print(len(out))
        layer_weights = []
        idx = 0
        for i in range(len(self.layer_sizes)-1):
            num_weights = self.layer_sizes[i+1] * self.layer_sizes[i]
            weight = out[idx:idx+num_weights].view(self.layer_sizes[i+1], self.layer_sizes[i]).detach()
            layer_weights.append(weight)
            idx += num_weights   

        preds = []
        for i in range(len(self.layers)):
            layer_weights[i] = layer_weights[i] -  self.layers[i].weight
            for k in range(layer_weights[i].shape[0]):
                for j  in range(layer_weights[i].shape[1]):
                    preds.append(round((-layer_weights[i][k][j]/shift).item()))
        print(len(preds))
        return preds



    def get_weights(self):
        """Get current weights as a single vector."""
        return torch.cat([layer.weight.view(-1) for layer in self.layers]).detach()

    def get_weight_statistics(self):
        """Calculate statistics of current weights."""
        weights = self.get_weights()
        return {
            'mean': weights.mean().item(),
            'std': weights.std().item(),
            'min': weights.min().item(),
            'max': weights.max().item(),
            'median': weights.median().item(),
            'q1': weights.quantile(0.25).item(),
            'q3': weights.quantile(0.75).item()
        }

    def total_parameters(self):
        """Calculate total number of parameters."""
        return sum(p.numel() for p in self.parameters())

class ExperimentTracker:
    """Tracks and visualizes experimental results."""

    def __init__(self, base_dir='results'):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.results = []

    def add_result(self, result):
        """Add a result to the tracker."""
        self.results.append(result)

    def save_results(self):
        """Save all results to disk."""
        results_file = os.path.join(self.experiment_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
    def plot_convergence(self, network, title_suffix='', save=True):
        """Plot convergence curves."""
        plt.figure(figsize=(10, 6))
        plt.semilogy(network.history['losses'], marker='o')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Loss (log scale)')
        plt.title(f'Convergence Plot {title_suffix}')
        
        if save:
            figure_name = f"exponential_convergence{title_suffix}.png"
            plt.savefig(os.path.join(self.experiment_dir, figure_name), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def plot_weight_distribution(self, network, title_suffix='', save=True):
        """Plot weight distribution analysis."""
        weights = network.get_weights().numpy()

        plt.figure(figsize=(10, 6))
        plt.hist(weights, bins=50, density=True, color='blue', alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.title(f'Weight Distribution {title_suffix}')
        plt.xlabel('Weight Value')
        plt.ylabel('Density')

        # Add vertical line for mean
        plt.axvline(x=np.mean(weights), color='red', linestyle='--', alpha=0.5)

        if save:
            # Format filename to match paper's naming convention
            ortho_suffix = "with_ortho" if network.use_orthogonal_layer else "no_ortho"
            activation = network.activation_function.lower()
            k_value = title_suffix.split('k')[1].split('_')[0] if 'k' in title_suffix else "10000"
            b_value = title_suffix.split('b')[1].split('_')[0] if 'b' in title_suffix else "0.1"
            
            figure_name = f"weight_distribution_{activation}_k{k_value}_b{b_value}_{ortho_suffix}.png"
            plt.savefig(os.path.join(self.experiment_dir, figure_name), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def train_quine(network, 
                k, 
                shift, 
                max_iterations=25,
                tolerance=1e-10):
    """
    Train network until convergence or max iterations reached.

    Returns:
        network: Trained network
        training_time: Total training time
        success: Whether training succeeded
    """
    start_time = time.time()
    success = False

    for iteration in range(max_iterations):
        current_weights = network.get_weights()
        network.regenerate(contraction=k, shift=shift)
        new_weights = network.get_weights()
        loss = ((current_weights - new_weights)**2).mean()

        # Update history
        network.history['losses'].append(loss.item())
        network.history['weight_stats'].append(network.get_weight_statistics())

        logging.info(f"Iteration {iteration+1}, Loss: {loss.item():.10f}")

        if torch.isnan(loss) or torch.isinf(loss) or loss > 1e6:
            logging.warning(f"Training failed at iteration {iteration+1}")
            break

        if loss.item() < tolerance:
            logging.info(f"Converged after {iteration+1} iterations!")
            success = True
            break
    else:
        logging.info("Did not converge within max iterations, increasing k")

    if not success:
        logging.info("Increasing contraction factor k and retrying with reinitialized network...")
        network = QuineNetwork(
            layer_sizes=network.layer_sizes,
            activation_function=network.activation_function,
            weight_init_method=network.weight_init_method,
            scale_factor=network.scale_factor,
            use_orthogonal_layer=network.use_orthogonal_layer
        )
        network, training_time, success, k = train_quine(network, k*10, shift, max_iterations, tolerance)
        return network, training_time, success, k

    training_time = time.time() - start_time
    return network, training_time, success, k

def run_experiments():
    """Run comprehensive experiments and generate results."""
    tracker = ExperimentTracker()

    # Experiment configurations matching the paper
    configs = [
        {'k': 1e4, 'shift': 0.1, 'activation': 'hardtanh_plus_x'},
        {'k': 1e5, 'shift': 0.1, 'activation': 'tanh'},
        {'k': 1e5, 'shift': 0.1, 'activation': 'relu_plus_x'},
        {'k': 1e5, 'shift': 0.2, 'activation': 'sin'},
    ]

    # Run experiments both with and without orthogonal layer
    for use_ortho in [False, True]:
        for config in configs:
            logging.info(f"\nRunning experiment with config: {config}, orthogonal layer: {use_ortho}")

            # Create network
            network = QuineNetwork(
                layer_sizes=[20, 512, 512, 512, 1],
                activation_function=config['activation'],
                weight_init_method='orthogonal',
                scale_factor=0.01,
                use_orthogonal_layer=use_ortho
            )

            # Train network
            network, training_time, success, k = train_quine(
                network, 
                k=config['k'], 
                shift=config['shift']
            )

            title_suffix = f"_{config['activation']}_k{int(config['k'])}_b{config['shift']}"
            if use_ortho:
                title_suffix += "_with_ortho"
            else:
                title_suffix += "_no_ortho"

            if success:
                # measure accuracy on the dataset
                measure_performance(network,shift_target,k,config['shift'])
                # Generate visualizations
                tracker.plot_convergence(network, title_suffix)
                tracker.plot_weight_distribution(network, title_suffix)

                # Record results
                result = {
                    'config': config,
                    'use_orthogonal_layer': use_ortho,
                    'training_time': training_time,
                    'iterations': len(network.history['losses']),
                    'final_loss': network.history['losses'][-1],
                    'success': success,
                    'total_parameters': network.total_parameters(),
                    'weight_statistics': network.history['weight_stats'][-1]
                }
                tracker.add_result(result)
            else:
                logging.warning(f"Training failed for config: {config}")
                result = {'config': config,
                    'use_orthogonal_layer': use_ortho,
                    'success': False
                }
                tracker.add_result(result)

    # Save all results
    tracker.save_results()

def run_ablation_study():
    """Run additional experiment with zero shift to demonstrate importance of non-zero shift."""
    tracker = ExperimentTracker()
    
    # Configuration for zero-shift experiment
    config = {'k': 1e4, 'shift': 0.0, 'activation': 'hardtanh'}
    
    network = QuineNetwork(
        layer_sizes=[20, 512, 512, 512, 1],
        activation_function=config['activation'],
        weight_init_method='orthogonal',
        scale_factor=0.01,
        use_orthogonal_layer=False
    )

    network, training_time, success, k = train_quine(
        network, 
        k=config['k'], 
        shift=config['shift']
    )

    title_suffix = f"_{config['activation']}_k{int(config['k'])}_b{config['shift']}_no_ortho"

    if success:
        tracker.plot_convergence(network, title_suffix)
        tracker.plot_weight_distribution(network, title_suffix)

        result = {
            'config': config,
            'use_orthogonal_layer': False,
            'training_time': training_time,
            'iterations': len(network.history['losses']),
            'final_loss': network.history['losses'][-1],
            'success': success,
            'total_parameters': network.total_parameters(),
            'weight_statistics': network.history['weight_stats'][-1]
        }
        tracker.add_result(result)
    
    tracker.save_results()

def main():
    """Main execution function."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Configure logging
    logging.info("Starting experiments...")

    try:
        # Run main experiments with and without orthogonal layer
        run_experiments()
        
        # Run ablation study with zero shift
        run_ablation_study()
        
        logging.info("All experiments completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during experiments: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()