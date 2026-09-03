# Bike67

A genetic algorithm that evolves UCI-legal road bicycle frame geometry to minimize power required at 45 km/h, using only the UCI rulebook as a constraint and no starting assumptions about what a bike frame should look like.

## What it does

The GA optimizes 19 frame parameters (tube angles and lengths, wheel diameters, handlebar geometry, and more) against a physics-based power model, while every candidate design is checked against real UCI legality constraints (dimensional limits, tube aspect ratios, weight minimums, and so on). Invalid designs get rejected before they're ever evaluated.

Stack:
- DEAP for the genetic algorithm
- Gymnasium as the simulation environment wrapper
- The Martin et al. (1998) validated road cycling power model for the physics

## Running it

```bash
pip install -r requirements.txt
python main.py                    # 100 generations, population 50
python main.py --generations 1000 --population 200
python main.py --seed 42          # reproducible run
```

Outputs winning_frame.json (the best design's full parameter set), winning_frame.png (a side-profile plot), and convergence.png (power vs generation).

## What I found

Across every run I've done, the algorithm consistently converges on asymmetric front and rear wheel sizes, typically a smaller front wheel around 550mm and a larger rear wheel between 620 and 700mm, rather than the matched wheel sizes every UCI road bike actually uses.

I don't think this means I've discovered something real teams have missed. It's more likely a limitation of the model. the fitness function only scores aerodynamic drag, rolling resistance, and weight. It doesn't account for the real reasons pro teams use matched wheels, like spare wheel logistics during a race, handling and rider familiarity, or manufacturing cost, none of which are captured in the simulation. It's a good example of how a genetic algorithm will happily exploit whatever your fitness function actually rewards, even if that's not quite the same as what you meant to optimize for. That gap is the most interesting part of the result, more than the wheel sizes themselves.

A few other things it consistently converges on: all four main tubes get pushed to the UCI's maximum legal aero section ratio at the same time, where human designers tend to optimize one or two tubes and compromise on the rest, and a noticeably steeper, more aggressive riding position than typical road bike geometry.

## Limitations

This is a simplified physics model, not a full CFD simulation, so the absolute numbers shouldn't be taken as production-accurate. The constraint set covers the UCI rules I implemented, not the complete rulebook. And as above, the surprising results likely say more about what the fitness function does and doesn't capture than about a genuine gap in real-world bike design.

## Files

- main.py, CLI entry point
- ga_optimizer.py, the DEAP evolution loop
- physics.py, the power and drag model
- constraints.py, UCI legality checks
- bike_env.py, the Gymnasium environment wrapper
- visualization.py, plotting

## License

MIT
