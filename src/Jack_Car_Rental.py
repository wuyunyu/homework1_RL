from calculate_value import *
from figure import *

# + 代表第一处移动到第二处
# - 代表第二处移动到第一处
actions = np.arange(-max_move_num, max_move_num + 1)  # 动作空间
value = np.zeros((max_car_num + 1, max_car_num + 1))  # 价值函数
policy = np.zeros(value.shape, dtype=int) # 策略
init_trans_prob()  # 初始化状态转移概率矩阵
# 策略迭代方法
def policy_iteration():
    iteration = 0
    while True:
    # 进行策略评估
        while True:
            old_value = value.copy()
            # 遍历所有状态
            for i in range(max_car_num + 1):
                for j in range(max_car_num + 1):
                    new_state_value = value_update([i, j], policy[i, j], value)
                    value[i, j] = new_state_value
            max_value_change = abs(old_value - value).max()
            # print(f'max value change: {max_value_change}')
            if max_value_change < 1e-4:
                break
        # 策略改进
        policy_stable = True
        for i in range(max_car_num + 1):
            for j in range(max_car_num + 1):
                old_action = policy[i, j]
                action_value = []
                # 遍历动作空间
                for action in actions:
                    if -j <= action <= i:  # valid action
                        action_value.append(value_update([i, j], action, value))
                    else:
                        action_value.append(-np.inf)
                action_value = np.array(action_value)
                # 贪婪选择，选择价值函数最大的动作
                new_action = actions[np.where(action_value == action_value.max())[0]]
                policy[i, j] = np.random.choice(new_action)
                if policy_stable and (old_action not in new_action):
                    policy_stable = False
        iteration += 1
        print('iteration: {}, policy stable {}'.format(iteration, policy_stable))
        draw_fig(value, policy, iteration)
        if policy_stable:
            break
def value_iteration():
    iteration = 0
    while True:
        delta = 0
        policy_stable = True
        # 遍历所有状态
        for i in range(max_car_num + 1):
            for j in range(max_car_num + 1):
                old_action = policy[i, j]
                old_value = value[i, j]
                action_value = []
                # 遍历动作空间
                for action in actions:
                    if -j <= action <= i:  # valid action
                        action_value.append(value_update([i, j], action, value))
                    else:
                        action_value.append(-np.inf)
                action_value = np.array(action_value)
                new_action = actions[np.where(action_value == action_value.max())[0]]
                policy[i, j] = np.random.choice(new_action)
                if policy_stable and (old_action not in new_action):
                    policy_stable = False
                new_value = action_value.max()
                value[i, j] = new_value
                delta = max(delta, abs(old_value - new_value))
        if policy_stable == True and delta < 0.1:
            break
        iteration += 1
        print('iteration: {}, delta: {}'.format(iteration, delta))
        draw_fig(value, policy, iteration)
policy_iteration()
# value_iteration()
