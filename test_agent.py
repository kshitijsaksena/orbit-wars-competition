from kaggle_environments import make
import main as my_agent
import sys

def run_tournament(num_episodes=10, mode="starter"):
    print(f"Running tournament against {mode} bot...")
    wins = 0
    draws = 0
    losses = 0
    
    for i in range(num_episodes):
        # We alternate player orders to ensure fairness
        players = [my_agent.agent, mode] if i % 2 == 0 else [mode, my_agent.agent]
        env = make("orbit_wars", debug=False)
        
        # Run episode
        env.run(players)
        
        # Save replay of the first game
        if i == 0:
            html = env.render(mode="html", width=800, height=600)
            filename = f"replay_{mode}.html"
            with open(filename, "w") as file:
                file.write(html)
            print(f"Saved first episode replay to {filename}")
            
        # Determine reward
        last_state = env.steps[-1]
        
        # The agent's reward is at index corresponding to its position
        agent_idx = 0 if i % 2 == 0 else 1
        agent_reward = last_state[agent_idx].reward
        
        if agent_reward == 1:
            wins += 1
            print(f"Episode {i+1}: WIN")
        elif agent_reward == 0:
            draws += 1
            print(f"Episode {i+1}: DRAW")
        else:
            losses += 1
            print(f"Episode {i+1}: LOSS")
            
    win_rate = (wins / num_episodes) * 100
    print(f"\nTournament complete against {mode}!")
    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    return win_rate

if __name__ == "__main__":
    # Test against starter bot
    starter_win_rate = run_tournament(10, "starter")
    
    # Test against random bot
    random_win_rate = run_tournament(10, "random")
    
    if starter_win_rate >= 80.0 and random_win_rate >= 90.0:
        print("\nAll verification tests passed successfully!")
        sys.exit(0)
    else:
        print("\nVerification failed: Win rate was too low.")
        sys.exit(1)
