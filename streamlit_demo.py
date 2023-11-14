import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as D
from matplotlib.animation import FuncAnimation

# Your Metropolis Hastings function
# Your Metropolis Hastings function
def metropolis_hastings(log_likelihood, initial, num_samples, burn_in, scale=1., case=1):
    samples = [initial]
    proposed_samples = []
    current = initial
    acceptance_rate=0
    for i in range(num_samples+burn_in):
        if case==1:
            normal_dist = D.Normal(current, scale)
        else:
            normal_dist = D.MultivariateNormal(current, scale * torch.eye(2))
        candidate = normal_dist.sample()
        proposed_samples.append((candidate, False))  # Initially mark all proposed samples as rejected
        log_alpha = log_likelihood(candidate) - log_likelihood(current)
        uniform_dist = D.Uniform(0, 1)
        if torch.log(uniform_dist.sample()) < log_alpha:
            current = candidate
            acceptance_rate+=1
            proposed_samples[-1] = (candidate, True)  # Mark accepted samples
        samples.append(current)

    return torch.stack(samples[burn_in:]), proposed_samples, acceptance_rate/(num_samples+burn_in)


# Streamlit app
def main():
    st.title("Metropolis Hastings Random Walk Demo")

    # Add a selectbox to the sidebar:
    add_selectbox = st.sidebar.selectbox(
        'What type of random walk would you like to see?',
        ('1D Random Walk', '2D Random Walk')
    )

    if add_selectbox == '1D Random Walk':
        st.header("1D Random Walk")

        # Dropdown for distribution selection
        distribution = st.selectbox(
            'Which distribution would you like to use?',
            ('Normal Distribution', 'Mixture of Gaussian')
        )

        # Number input for number of samples and sigma
        num_samples = st.number_input('Enter the number of samples', min_value=1, value=100, step=1)
        sigma = st.slider('Enter the sigma of proposal distribution', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        option = st.selectbox(
            'Which form of output do you want?',
            ('Static plot', 'GIF')
        )
        # Define the distributions
        if distribution == 'Normal Distribution':
            dist = D.Normal(torch.tensor([3.0]), torch.tensor([2.0]))
        else:
            dist1 = D.Normal(torch.tensor([2.0]), torch.tensor([1.0]))
            dist2 = D.Normal(torch.tensor([-2.0]), torch.tensor([1.0]))
            dist = D.MixtureSameFamily(D.Categorical(torch.tensor([0.3, 0.7])), D.Normal(torch.cat([dist1.mean, dist2.mean]), torch.cat([dist1.stddev, dist2.stddev])))

        # Button to run Metropolis Hastings
        initial = dist.sample()
        samples, proposed_samples , acceptance_rate = metropolis_hastings(dist.log_prob, initial, num_samples, burn_in=0, scale=sigma)
        
        if option=='Static plot':
            # Plotting after all sampling has been done
            fig, ax = plt.subplots()
            samples_np = samples.numpy()
            x=np.linspace(-5, 5, 100)
            y=dist.log_prob(torch.tensor(x)).exp().numpy()
            ax.plot(x, y, color='green', label='True Distribution', linewidth=3)
            for i in range(0, len(samples_np), int(len(samples_np)/10)):
                color = i / len(samples_np)  # Color changes with iteration
                sns.kdeplot(samples_np[:i+int(len(samples_np)/10)], fill=True, color=plt.cm.plasma(color, alpha=0.5))  # KDE plot with colormap
            ax.scatter(samples_np, np.zeros_like(samples_np), color=plt.cm.plasma(color))  # Scatter plot of samples with colormap
            ax.set_title("1D Random Walk")
            fig.colorbar(plt.cm.ScalarMappable(cmap='plasma',norm=plt.Normalize(0, len(samples_np))), ax=ax)
            ax.legend()
            st.pyplot(fig)
        else:
            i = st.slider('Step', min_value=0, max_value=len(samples)-1, value=0)

            # Plotting the current sample and the KDE of all samples up to the current iteration
            fig, ax = plt.subplots()
            color = 'red' if i > 0 and samples[i] == samples[i-1] else 'green'  # Update color based on acceptance
            ax.scatter(samples[:i+1].numpy(), np.zeros_like(samples[:i+1].numpy()), color=color)  # Scatter plot of current sample
            sns.kdeplot(samples[:i+1].numpy(), fill=True, color='blue', alpha=0.5, ax=ax)  # KDE plot of all samples up to current iteration
            x=np.linspace(-5, 5, 100)
            y=dist.log_prob(torch.tensor(x)).exp().numpy()
            ax.plot(x, y, color='green', label='True Distribution', linewidth=3)
            ax.set_title("1D Random Walk")
            ax.legend()
            st.pyplot(fig)

    else:
        st.header("2D Random Walk")

        # Dropdown for distribution selection
        distribution = st.selectbox(
            'Which distribution would you like to use?',
            ('Normal Distribution', 'Mixture of Gaussian')
        )

        # Number input for number of samples and sigma
        num_samples = st.number_input('Enter the number of samples', min_value=1, value=100, step=1)
        sigma = st.slider('Enter the sigma of proposal distribution', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        option = st.selectbox(
            'Which form of output do you want?',
            ('Static plot', 'GIF')
        )
        dist=None
        # Define the distributions
        if distribution == 'Normal Distribution':
            dist = D.MultivariateNormal(torch.tensor([1.0, 1.0]), torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
        else:
            mix = D.Categorical(torch.tensor([0.5, 0.5]))
            comp = D.MultivariateNormal(torch.tensor([[-1.0, -1.0], [1.0, 1.0]]), torch.eye(2).expand(2, -1, -1))
            dist = D.MixtureSameFamily(mix, comp)
        # Button to run Metropolis Hastings
        initial = dist.sample()
        samples, proposed_samples, acceptance_rate = metropolis_hastings(dist.log_prob, initial, num_samples, burn_in=0, scale=sigma*torch.eye(2),case=2)

        if option=='Static plot':
            # Plotting after all sampling has been done
            fig = plt.figure(figsize=(10, 10))
            gs = fig.add_gridspec(4, 4)
            ax1 = fig.add_subplot(gs[1:4, 0:3])
            ax2 = fig.add_subplot(gs[0, 0:3])
            ax3 = fig.add_subplot(gs[1:4, 3])

            samples_np = samples.numpy()
            for i in range(0, len(samples_np), int(len(samples_np)/10)):
                color = i / len(samples_np)  # Color changes with iteration
                #sns.kdeplot(x=samples_np[:i+20, 0], y=samples_np[:i+20, 1], fill=True, color=plt.cm.cool(color, alpha=0.5), ax=ax1)  # KDE plot with colormap
                sns.kdeplot(x=samples_np[:i+int(len(samples_np)/10), 0], fill=True, color=plt.cm.plasma(color, alpha=0.5), ax=ax2)  # Marginal KDE plot for x
                sns.kdeplot(x=samples_np[:i+int(len(samples_np)/10), 1], fill=True, color=plt.cm.plasma(color, alpha=0.5), ax=ax3, vertical=True)  # Marginal KDE plot for y
            ax1.scatter(samples_np[:, 0], samples_np[:, 1], color=plt.cm.plasma(color))  # Scatter plot of samples with colormap
            x=np.linspace(-5, 5, 100)
            y=np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y
            ax1.contour(X, Y, dist.log_prob(torch.tensor(pos)).exp().numpy(), cmap='cool')
            ax1.set_title("2D Random Walk")
            fig.colorbar(plt.cm.ScalarMappable(cmap='plasma',norm=plt.Normalize(0, len(samples_np))), ax=[ax1, ax2, ax3])
            st.pyplot(fig)
        else:
            i = st.slider('Step', min_value=0, max_value=len(samples)-1, value=0)

            # Plotting the current sample and the KDE of all samples up to the current iteration
            fig = plt.figure(figsize=(10, 10))
            gs = fig.add_gridspec(4, 4)
            ax1 = fig.add_subplot(gs[1:4, 0:3])
            ax2 = fig.add_subplot(gs[0, 0:3])
            ax3 = fig.add_subplot(gs[1:4, 3])
            ax1.scatter(samples[:i+1][:,0],samples[:i+1][:,1])  # Scatter plot of current sample
            sns.kdeplot(x=samples[:i+1,0], fill=True, color='blue', alpha=0.5, ax=ax2)  # KDE plot of all samples up to current iteration
            sns.kdeplot(x=samples[:i+1,1], fill=True, color='blue', alpha=0.5, ax=ax3, vertical=True)  # KDE plot of all samples up to current iteration
            x=np.linspace(-5, 5, 100)
            y=np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y
            ax1.contour(X, Y, dist.log_prob(torch.tensor(pos)).exp().numpy(), cmap='cool')
            ax1.set_title("2D Random Walk")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
