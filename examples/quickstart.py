"""Minimal quickstart example for Katai Browser Agent."""

from katai import KataiAgent

def main():
    # Initializes KataiAgent with the fine-tuned v10s model (322M mmBERT-base)
    agent = KataiAgent(checkpoint="v10s", headless=False)

    url = "https://en.wikipedia.org/wiki/Main_Page"
    goal = "Search Wikipedia for 'Python programming language' and open the article."

    print(f"Starting agent on {url} with goal: {goal}")
    agent.start(url, goal)

    step = 0
    while agent.state["status"] not in ("done", "blocked") and step < 20:
        step += 1
        agent.step()
        last_action = agent.state["history"][-1] if agent.state["history"] else {}
        print(f"Step {step:2d} | Status: {agent.state['status']:7s} | Action: {last_action.get('action')}")

    print(f"Finished with status: {agent.state['status'].upper()} in {agent.state['elapsed_ms']} ms")
    agent.close()

if __name__ == "__main__":
    main()
