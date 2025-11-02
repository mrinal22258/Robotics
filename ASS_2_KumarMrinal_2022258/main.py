import matplotlib.pyplot as plt
from apf_calculations import (
    START, GOAL, OBSTACLES,
    KA, KR, N0, GAMMA,
    generate_trajectory_gradient_flow,
    simulate_bicycle_following,
    U_total
)
from plotting_animation import (
    plot_configuration_space, plot_potential_field, create_animation
)

def run_all():
    print('--- Artificial Potential Field (Gradient Flow) ---')

    print('Generating trajectories...')
    traj_parabolic = generate_trajectory_gradient_flow(START, GOAL, OBSTACLES, ka=KA, kr_array=KR, n0_array=N0, gamma=GAMMA, att_type='parabolic')
    traj_conical = generate_trajectory_gradient_flow(START, GOAL, OBSTACLES, ka=KA, kr_array=KR, n0_array=N0, gamma=GAMMA, att_type='conical')

    print('Plotting potential fields...')
    fig1, _ = plot_potential_field(OBSTACLES, GOAL, KA, KR, N0, GAMMA, U_total, att_type='parabolic')
    fig1.savefig('potential_field_parabolic.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)

    fig2, _ = plot_potential_field(OBSTACLES, GOAL, KA, KR, N0, GAMMA, U_total, att_type='conical')
    fig2.savefig('potential_field_conical.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print('Saving trajectory plots...')
    f1, _ = plot_configuration_space(OBSTACLES, START, GOAL, traj_parabolic, 'Parabolic Attractive Potential')
    f1.savefig('trajectory_parabolic.png', dpi=150, bbox_inches='tight')
    plt.close(f1)

    f2, _ = plot_configuration_space(OBSTACLES, START, GOAL, traj_conical, 'Conical Attractive Potential')
    f2.savefig('trajectory_conical.png', dpi=150, bbox_inches='tight')
    plt.close(f2)

    print('Creating animations...')
    states_parabolic = simulate_bicycle_following(traj_parabolic)
    states_conical = simulate_bicycle_following(traj_conical)

    # animate
    create_animation(states_parabolic, traj_parabolic, OBSTACLES, 'apf_animation_parabolic.gif')
    create_animation(states_conical, traj_conical, OBSTACLES, 'apf_animation_conical.gif')


    print('\nAll plots and animations saved successfully.')

if __name__ == '__main__':
    run_all()
