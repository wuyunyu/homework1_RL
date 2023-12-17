from calculate_value import *
from figure import *

# + 代表第一处移动到第二处
# - 代表第二处移动到第一处
actions = np.arange(-max_move_num, max_move_num + 1)  # 动作空间
value = np.zeros((max_car_num + 1, max_car_num + 1))  # 价值函数
policy = np.zeros(value.shape, dtype=int) # 策略
init_trans_prob()  # 初始化状态转移概率矩阵

# policy iteration
def policy_iteration():
    iteration = 0
    while True:
    # policy evaluattion
        while True:
            old_value = value.copy()
            for i in range(max_car_num + 1):
                for j in range(max_car_num + 1):
                    new_state_value = value_update([i, j], policy[i, j], value)
                    value[i, j] = new_state_value
            max_value_change = abs(old_value - value).max()
            # print(f'max value change: {max_value_change}')
            if max_value_change < 1e-4:
                break
        # policy improvement
        policy_stable = True
        for i in range(max_car_num + 1):
            for j in range(max_car_num + 1):
                old_action = policy[i, j]
                action_value = []
                for action in actions:
                    if -j <= action <= i:  # valid action
                        action_value.append(value_update([i, j], action, value))
                    else:
                        action_value.append(-np.inf)
                action_value = np.array(action_value)
                # greedy policy
                new_action = actions[np.where(action_value == action_value.max())[0]]
                policy[i, j] = np.random.choice(new_action)
                if policy_stable and (old_action not in new_action):
                    policy_stable = False
        iteration += 1
        print('iteration: {}, policy stable {}'.format(iteration, policy_stable))
        draw_fig(value, policy, iteration, 0)
        if policy_stable:
            break

# value iteration
def value_iteration():
    iteration = 0
    while True:
        delta = 0
        for i in range(max_car_num + 1):
            for j in range(max_car_num + 1):
                old_value = value[i, j]
                action_value = []
                for action in actions:
                    if -j <= action <= i:  # valid action
                        action_value.append(value_update([i, j], action, value))
                    else:
                        action_value.append(-np.inf)
                action_value = np.array(action_value)
                # greedy policy
                # policy update
                new_action = actions[np.where(action_value == action_value.max())[0]]
                policy[i, j] = np.random.choice(new_action)
                # value update
                new_value = action_value.max()
                value[i, j] = new_value
                delta = max(delta, abs(old_value - new_value))
        iteration += 1
        print('iteration: {}, delta: {}'.format(iteration, delta))
        draw_fig(value, policy, iteration, 1)
        if delta < 0.1:
            break

policy_iteration()
# value_iteration()
