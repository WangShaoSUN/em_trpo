import os

import numpy as np
import torch
from torch.optim import Adam

from slac.buffer import ReplayBuffer
from slac.network import GaussianPolicy, LatentModel, TwinnedQNetwork
from slac.utils import create_feature_actions, grad_false, soft_update

FLOAT_EPSILON = 1e-8
def tile(x, n):
    return x[None].repeat([n] + [1] * len(x.shape))


def merge_first_two_dims(x):
    return x.view(x.shape[0] * x.shape[1], *x.shape[2:])
class SlacAlgorithm:
    """
    Stochactic Latent Actor-Critic(SLAC).

    Paper: https://arxiv.org/abs/1907.00953
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        action_repeat,
        device,
        seed,
        gamma=0.99,
        batch_size_sac=256,
        batch_size_latent=32,
        buffer_size=10 ** 5,
        num_sequences=8,
        lr_sac=3e-4,
        lr_latent=1e-4,
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
        tau=5e-3,
        num_samples=20, epsilon=1e-1, epsilon_penalty=1e-3,
        epsilon_mean=1e-3, epsilon_std=1e-6, initial_log_temperature=1.,
        initial_log_alpha_mean=1., initial_log_alpha_std=10.,
        min_log_dual=-18., per_dim_constraining=True, action_penalization=True,
        actor_optimizer=None, dual_optimizer=None, gradient_clip=0
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Replay buffer.
        self.buffer = ReplayBuffer(buffer_size, num_sequences, state_shape, action_shape, device)

        # Networks.
        self.actor = GaussianPolicy(action_shape, num_sequences, feature_dim, hidden_units).to(device)
        self.actor_target = GaussianPolicy(action_shape, num_sequences, feature_dim, hidden_units).to(device)
        soft_update(self.actor_target, self.actor, 1.0)


        self.critic = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units).to(device)
        self.critic_target = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units).to(device)
        self.latent = LatentModel(state_shape, action_shape, feature_dim, z1_dim, z2_dim, hidden_units).to(device)
        soft_update(self.critic_target, self.critic, 1.0)
        grad_false(self.critic_target)

        # Target entropy is -|A|.
        self.target_entropy = -float(action_shape[0])
        # We optimize log(alpha) because alpha is always bigger than 0.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

        # Optimizers.
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_sac)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_sac)
        self.optim_alpha = Adam([self.log_alpha], lr=lr_sac)
        self.optim_latent = Adam(self.latent.parameters(), lr=lr_latent)

        self.learning_steps_sac = 0
        self.learning_steps_latent = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_repeat = action_repeat
        self.device = device
        self.gamma = gamma
        self.batch_size_sac = batch_size_sac
        self.batch_size_latent = batch_size_latent
        self.num_sequences = num_sequences
        self.tau = tau

        # JIT compile to speed up.
        fake_feature = torch.empty(1, num_sequences + 1, feature_dim, device=device)
        fake_action = torch.empty(1, num_sequences, action_shape[0], device=device)
        self.create_feature_actions = torch.jit.trace(create_feature_actions, (fake_feature, fake_action))

        self.num_samples = num_samples
        self.epsilon = epsilon
        self.epsilon_mean = epsilon_mean
        self.epsilon_std = epsilon_std
        self.initial_log_temperature = initial_log_temperature
        self.initial_log_alpha_mean = initial_log_alpha_mean
        self.initial_log_alpha_std = initial_log_alpha_std
        self.min_log_dual = torch.as_tensor(min_log_dual, dtype=torch.float32)
        self.action_penalization = action_penalization
        self.epsilon_penalty = epsilon_penalty
        self.per_dim_constraining = per_dim_constraining
        self.gradient_clip = gradient_clip
        self.dual_optimizer =  (
            lambda params: torch.optim.Adam(params, lr=1e-2))

        self.dual_variables = []
        self.log_temperature = torch.nn.Parameter(torch.as_tensor(
            [self.initial_log_temperature], dtype=torch.float32))
        self.dual_variables.append(self.log_temperature)
        shape = [action_shape[0]] if self.per_dim_constraining else [1]
        self.log_alpha_mean = torch.nn.Parameter(torch.full(
            shape, self.initial_log_alpha_mean, dtype=torch.float32))
        self.dual_variables.append(self.log_alpha_mean)
        self.log_alpha_std = torch.nn.Parameter(torch.full(
            shape, self.initial_log_alpha_std, dtype=torch.float32))
        self.dual_variables.append(self.log_alpha_std)
        if self.action_penalization:
            self.log_penalty_temperature = torch.nn.Parameter(torch.as_tensor(
                [self.initial_log_temperature], dtype=torch.float32))
            self.dual_variables.append(self.log_penalty_temperature)
        self.dual_optimizer = self.dual_optimizer(self.dual_variables)

    def preprocess(self, ob):
        state = torch.tensor(ob.state, dtype=torch.uint8, device=self.device).float().div_(255.0)
        with torch.no_grad():
            feature = self.latent.encoder(state).view(1, -1)
        action = torch.tensor(ob.action, dtype=torch.float, device=self.device)
        feature_action = torch.cat([feature, action], dim=1)
        return feature_action

    def explore(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor.sample(feature_action)[0]
        return action.cpu().numpy()[0]

    def exploit(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor(feature_action)
        return action.cpu().numpy()[0]

    def step(self, env, ob, t, is_random):
        t += 1

        if is_random:
            action = env.action_space.sample()
        else:
            action = self.explore(ob)

        state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        ob.append(state, action)
        self.buffer.append(action, reward, mask, state, done)

        if done:
            t = 0
            state = env.reset()
            ob.reset_episode(state)
            self.buffer.reset_episode(state)

        return t

    def update_latent(self, writer):
        self.learning_steps_latent += 1
        state_, action_, reward_, done_ = self.buffer.sample_latent(self.batch_size_latent)
        loss_kld, loss_image, loss_reward = self.latent.calculate_loss(state_, action_, reward_, done_)

        self.optim_latent.zero_grad()
        (loss_kld + loss_image + loss_reward).backward()
        self.optim_latent.step()

        if self.learning_steps_latent % 1000 == 0:
            pass
            # writer.add_scalar("loss/kld", loss_kld.item(), self.learning_steps_latent)
            # writer.add_scalar("loss/reward", loss_reward.item(), self.learning_steps_latent)
            # writer.add_scalar("loss/image", loss_image.item(), self.learning_steps_latent)

    def update_sac(self, writer):
        self.learning_steps_sac += 1
        state_, action_, reward, done = self.buffer.sample_sac(self.batch_size_sac)
        z, next_z, action, feature_action, next_feature_action = self.prepare_batch(state_, action_)

        self.update_critic(z, next_z, action, next_feature_action, reward, done, writer)
        self.update_actor(z, feature_action, writer)
        soft_update(self.critic_target, self.critic, self.tau)

    def prepare_batch(self, state_, action_):
        with torch.no_grad():
            # f(1:t+1)
            feature_ = self.latent.encoder(state_)
            # z(1:t+1)
            z_ = torch.cat(self.latent.sample_posterior(feature_, action_)[2:], dim=-1)

        # z(t), z(t+1)
        z, next_z = z_[:, -2], z_[:, -1]
        # a(t)
        action = action_[:, -1]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_action, next_feature_action = self.create_feature_actions(feature_, action_)

        return z, next_z, action, feature_action, next_feature_action

    def update_critic(self, z, next_z, action, next_feature_action, reward, done, writer):
        curr_q1, curr_q2 = self.critic(z, action)
        with torch.no_grad():
            next_action, log_pi = self.actor.sample(next_feature_action)
            next_q1, next_q2 = self.critic_target(next_z, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * log_pi
        target_q = reward + (1.0 - done) * self.gamma * next_q
        loss_critic = (curr_q1 - target_q).pow_(2).mean() + (curr_q2 - target_q).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        self.optim_critic.step()

        if self.learning_steps_sac % 1000 == 0:
            # writer.add_scalar("loss/critic", loss_critic.item(), self.learning_steps_sac)
            pass
    def update_actor(self, z, feature_action, writer):
        # print("feat_act:",feature_action.shape)
        def parametric_kl_and_dual_losses(kl, alpha, epsilon):
            kl_mean = kl.mean(dim=0).to(self.device)
            kl_loss = (alpha.detach().to(self.device) * kl_mean).sum()
            alpha_loss = (alpha.to(self.device) * (epsilon - kl_mean.detach())).sum()
            return kl_loss, alpha_loss

        def weights_and_temperature_loss(q_values, epsilon, temperature):
            tempered_q_values = q_values.detach() / temperature.to(self.device)
            # print(q_values)
            # print(q_values.shape)
            # print("tem:",temperature)
            # tempered_q_values = q_values/ temperature

            weights = torch.nn.functional.softmax(tempered_q_values, dim=0)
            weights = weights.detach()

            # Temperature loss (dual of the E-step).
            q_log_sum_exp = torch.logsumexp(tempered_q_values, dim=0)
            num_actions = torch.as_tensor(
                q_values.shape[0], dtype=torch.float32)
            log_num_actions = torch.log(num_actions)
            loss = (epsilon + (q_log_sum_exp).mean() - log_num_actions).to(self.device)
            loss = temperature.to(self.device) * loss

            return weights, loss

        def independent_normals(distribution_1, distribution_2=None):
            distribution_2 = distribution_2 or distribution_1
            return torch.distributions.independent.Independent(
                torch.distributions.normal.Normal(
                    distribution_1.mean, distribution_2.stddev), -1)

        with torch.no_grad():
            self.log_temperature.data.copy_(
                torch.max(self.min_log_dual, self.log_temperature))
            self.log_alpha_mean.data.copy_(
                torch.max(self.min_log_dual, self.log_alpha_mean))
            self.log_alpha_std.data.copy_(
                torch.max(self.min_log_dual, self.log_alpha_std))
            if self.action_penalization:
                self.log_penalty_temperature.data.copy_(torch.max(
                    self.min_log_dual, self.log_penalty_temperature))
            target_distributions=self.actor_target.sample_distribution(feature_action)
            actions=target_distributions.sample((self.num_samples,)).to(self.device)

            tiled_observations=tile(z, self.num_samples).to(self.device)
            flat_observations = merge_first_two_dims(tiled_observations)
            flat_actions = merge_first_two_dims(actions)
            values,_=self.critic_target(flat_observations,flat_actions)
            values = values.view(self.num_samples, -1).to(self.device)
            # print("val:",values.shape)
            assert isinstance(
                target_distributions, torch.distributions.normal.Normal)
            target_distributions = independent_normals(target_distributions)
        self.optim_actor.zero_grad()
        self.dual_optimizer.zero_grad()

        distributions=self.actor.sample_distribution(feature_action)
        distributions = independent_normals(distributions)

        temperature = torch.nn.functional.softplus(
            self.log_temperature) + FLOAT_EPSILON
        alpha_mean = torch.nn.functional.softplus(
            self.log_alpha_mean) + FLOAT_EPSILON
        alpha_std = torch.nn.functional.softplus(
            self.log_alpha_std) + FLOAT_EPSILON
        weights, temperature_loss = weights_and_temperature_loss(
            values, self.epsilon, temperature)

        if self.action_penalization:
            penalty_temperature = torch.nn.functional.softplus(
                self.log_penalty_temperature) + FLOAT_EPSILON
            diff_bounds = actions - torch.clamp(actions, -1, 1)
            action_bound_costs = -torch.norm(diff_bounds, dim=-1)
            penalty_weights, penalty_temperature_loss = \
                weights_and_temperature_loss(
                    action_bound_costs,
                    self.epsilon_penalty, penalty_temperature)
            weights += penalty_weights
            temperature_loss += penalty_temperature_loss

        fixed_std_distribution = independent_normals(
            distributions.base_dist, target_distributions.base_dist)
        fixed_mean_distribution = independent_normals(
            target_distributions.base_dist, distributions.base_dist)

        policy_mean_losses = (fixed_std_distribution.base_dist.log_prob(
            actions).sum(dim=-1) * weights).sum(dim=0)
        policy_mean_loss = -(policy_mean_losses).mean()
        policy_std_losses = (fixed_mean_distribution.base_dist.log_prob(
            actions).sum(dim=-1) * weights).sum(dim=0)
        policy_std_loss = -policy_std_losses.mean()

        if self.per_dim_constraining:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_std_distribution.base_dist)
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_mean_distribution.base_dist)
        else:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_std_distribution)
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_mean_distribution)

        kl_mean_loss, alpha_mean_loss = parametric_kl_and_dual_losses(
            kl_mean, alpha_mean, self.epsilon_mean)
        kl_std_loss, alpha_std_loss = parametric_kl_and_dual_losses(
            kl_std, alpha_std, self.epsilon_std)

        policy_loss = policy_mean_loss + policy_std_loss
        kl_loss = kl_mean_loss + kl_std_loss
        dual_loss = alpha_mean_loss + alpha_std_loss + temperature_loss
        loss = policy_loss + kl_loss + dual_loss

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.gradient_clip)
            torch.nn.utils.clip_grad_norm_(
                self.dual_variables, self.gradient_clip)
        self.optim_actor.step()
        self.dual_optimizer.step()
        # action, log_pi = self.actor.sample(feature_action)
        # q1, q2 = self.critic(z, action)
        # loss_actor = -torch.mean(torch.min(q1, q2) - self.alpha * log_pi)
        #
        # self.optim_actor.zero_grad()
        # loss_actor.backward(retain_graph=False)
        # self.optim_actor.step()
        #
        # with torch.no_grad():
        #     entropy = -log_pi.detach().mean()
        # loss_alpha = -self.log_alpha * (self.target_entropy - entropy)
        #
        # self.optim_alpha.zero_grad()
        # loss_alpha.backward(retain_graph=False)
        # self.optim_alpha.step()
        # with torch.no_grad():
        #     self.alpha = self.log_alpha.exp()

        if self.learning_steps_sac % 1000 == 0:
            pass
            # writer.add_scalar("loss/actor", loss_actor.item(), self.learning_steps_sac)
            # writer.add_scalar("loss/alpha", loss_alpha.item(), self.learning_steps_sac)
            # writer.add_scalar("stats/alpha", self.alpha.item(), self.learning_steps_sac)
            # writer.add_scalar("stats/entropy", entropy.item(), self.learning_steps_sac)

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # We don't save target network to reduce workloads.
        torch.save(self.latent.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
        torch.save(self.latent.state_dict(), os.path.join(save_dir, "latent.pth"))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
