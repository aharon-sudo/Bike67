"""
DEAP-based genetic algorithm for bicycle frame geometry optimisation.

Algorithm:
    Selection:  Tournament (size = 3)
    Crossover:  Two-point crossover  (cxTwoPoint)
    Mutation:   Gaussian perturbation per gene (mutGaussian)
    Elitism:    Best 5 % carried forward unchanged each generation

Initial population:
    50 individuals sampled uniformly within parameter bounds.
    The first 10 are deliberately "unoptimised box" designs —
    high stack, short reach, round tubes — so the GA must actively
    discover better solutions rather than starting near the optimum.

Genome encoding:
    A flat list of N_PARAMS = 15 floats in raw physical units.
    After every crossover/mutation step the genome is hard-clipped to its
    declared bounds (repair operator) to guarantee structural validity.
    UCI constraint violations are handled as soft penalties in the fitness
    function, so the GA naturally migrates toward the legal region.
"""

from __future__ import annotations

import random
import numpy as np
from typing import List, Tuple, Optional

from deap import base, creator, tools, algorithms

from physics import PARAM_NAMES, PARAM_BOUNDS, N_PARAMS, genome_to_dict
from constraints import repair_genome
from bike_env import evaluate_frame

# ── DEAP type registration (module-level, idempotent) ─────────────────────────
if not hasattr(creator, 'FitnessMin'):
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))  # minimise power (W)
if not hasattr(creator, 'Individual'):
    creator.create('Individual', list, fitness=creator.FitnessMin)


# ── Population factory ────────────────────────────────────────────────────────

def _random_genome() -> list:
    """Uniform random genome within declared bounds."""
    return [random.uniform(lo, hi) for lo, hi in PARAM_BOUNDS]


def _box_genome() -> list:
    """
    'Box' genome: UCI-legal but aerodynamically naive design.
    Represents the kind of frame a naive engineer might draw to satisfy
    the rules and nothing more:
        • Maximum stack (most upright, worst rider CdA)
        • Minimum reach (shortest cockpit extension)
        • Round tubes (worst aerodynamic section, aspect = 1.0)
        • Frame weight close to minimum legal (6.8 kg)
        • Other params randomly sampled within bounds
    """
    idx = {n: i for i, n in enumerate(PARAM_NAMES)}
    genome = _random_genome()
    genome[idx['stack']]              = PARAM_BOUNDS[idx['stack']][1]        # max
    genome[idx['reach']]              = PARAM_BOUNDS[idx['reach']][0]        # min
    genome[idx['down_tube_aspect']]   = PARAM_BOUNDS[idx['down_tube_aspect']][0]  # round
    genome[idx['seat_tube_aspect']]   = PARAM_BOUNDS[idx['seat_tube_aspect']][0]
    genome[idx['top_tube_aspect']]    = PARAM_BOUNDS[idx['top_tube_aspect']][0]
    genome[idx['chainstay_aspect']]   = PARAM_BOUNDS[idx['chainstay_aspect']][0]
    genome[idx['frame_weight']]       = 6.8 + random.uniform(0.0, 0.5)      # near UCI min
    return genome


def create_individual(box: bool = False) -> creator.Individual:
    genome = _box_genome() if box else _random_genome()
    ind = creator.Individual(genome)
    return ind


def create_initial_population(pop_size: int = 50, n_box: int = 10) -> List:
    """
    Create the initial population.

    Args:
        pop_size:  Total population size (50 per spec).
        n_box:     Number of 'box' individuals inserted to ensure
                   the GA has to work from an unoptimised start.

    Returns:
        List of DEAP Individual objects.
    """
    population = []
    for i in range(pop_size):
        use_box = (i < n_box)
        population.append(create_individual(box=use_box))
    return population


# ── Repair operator ────────────────────────────────────────────────────────────

def _repair(individual: creator.Individual) -> creator.Individual:
    """Clip all genes to their declared bounds in-place."""
    repaired = repair_genome(list(individual))
    individual[:] = repaired
    return individual


# ── Custom mutation: Gaussian with adaptive sigma ─────────────────────────────

def _adaptive_gaussian_mutate(
    individual: creator.Individual,
    sigma_frac: float = 0.20,
    indpb: float = 0.25,
) -> Tuple[creator.Individual]:
    """
    Gaussian mutation where sigma scales to each parameter's range.
    sigma = sigma_frac × (upper − lower) for each gene independently.

    Args:
        sigma_frac: fraction of parameter range to use as one-sigma width
        indpb:      probability of mutating each gene
    """
    for i, (lo, hi) in enumerate(PARAM_BOUNDS):
        if random.random() < indpb:
            sigma = sigma_frac * (hi - lo)
            individual[i] += random.gauss(0.0, sigma)
    _repair(individual)
    del individual.fitness.values     # mark fitness as stale
    return (individual,)


# ── DEAP toolbox construction ─────────────────────────────────────────────────

def build_toolbox(
    tournament_size: int = 3,
    cx_prob: float = 0.7,
    mut_sigma_frac: float = 0.20,
    mut_gene_prob: float = 0.25,
) -> base.Toolbox:
    """
    Assemble and return a configured DEAP Toolbox.

    Registered operators:
        evaluate  — penalised CdA fitness (lower = better)
        mate      — two-point crossover
        mutate    — adaptive Gaussian per-gene mutation
        select    — tournament selection
    """
    toolbox = base.Toolbox()

    # ── Evaluation ────────────────────────────────────────────────────────────
    toolbox.register('evaluate', evaluate_frame)   # returns (penalised_power_W,)

    # ── Crossover: two-point works well for physically-structured genomes ─────
    def _cx_with_repair(ind1, ind2):
        tools.cxTwoPoint(ind1, ind2)
        _repair(ind1)
        _repair(ind2)
        del ind1.fitness.values
        del ind2.fitness.values
        return ind1, ind2

    toolbox.register('mate', _cx_with_repair)

    # ── Mutation ──────────────────────────────────────────────────────────────
    toolbox.register(
        'mutate',
        _adaptive_gaussian_mutate,
        sigma_frac=mut_sigma_frac,
        indpb=mut_gene_prob,
    )

    # ── Selection: tournament ─────────────────────────────────────────────────
    toolbox.register('select', tools.selTournament, tournsize=tournament_size)

    return toolbox


# ── Generation statistics ─────────────────────────────────────────────────────

def build_stats() -> tools.Statistics:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register('min',  np.min)
    stats.register('mean', np.mean)
    stats.register('max',  np.max)
    stats.register('std',  np.std)
    return stats


# ── Main evolutionary loop ────────────────────────────────────────────────────

def run_evolution(
    n_generations: int = 1000,
    pop_size: int = 200,
    n_box: int = 10,
    cx_prob: float = 0.7,
    mut_prob: float = 0.3,
    elitism_frac: float = 0.05,
    tournament_size: int = 3,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List, tools.Logbook, creator.Individual]:
    """
    Run the full genetic algorithm.

    Args:
        n_generations:  Number of generations to evolve (100 per spec).
        pop_size:       Population size (50 per spec).
        n_box:          Unoptimised 'box' individuals in gen-0 population.
        cx_prob:        Crossover probability per pair.
        mut_prob:       Mutation probability per individual.
        elitism_frac:   Fraction of best individuals preserved each generation.
        tournament_size: Tournament selection group size.
        seed:           Random seed for reproducibility (None = random).
        verbose:        Print per-generation summary.

    Returns:
        (final_population, logbook, hall_of_fame[0])
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    toolbox = build_toolbox(tournament_size=tournament_size)
    stats   = build_stats()
    hof     = tools.HallOfFame(maxsize=1)   # track global best

    n_elite = max(1, int(pop_size * elitism_frac))

    # ── Initialise population ─────────────────────────────────────────────────
    population = create_initial_population(pop_size, n_box)

    # Evaluate initial fitnesses
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    hof.update(population)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'n_evals', 'min', 'mean', 'std', 'max']

    record = stats.compile(population)
    logbook.record(gen=0, n_evals=pop_size, **record)
    prev_best = hof[0].fitness.values[0]
    if verbose:
        _print_generation(0, record, hof[0])   # always print initial state

    # ── Generational loop ──────────────────────────────────────────────────────
    for gen in range(1, n_generations + 1):

        # Elite preservation: copy the top n_elite unchanged
        elite = tools.selBest(population, n_elite)
        elite = list(map(toolbox.clone, elite))

        # Tournament selection for offspring pool
        offspring = toolbox.select(population, pop_size - n_elite)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover — explicit index loop handles odd-length offspring correctly
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_prob:
                toolbox.mate(offspring[i], offspring[i + 1])

        # Mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)

        # Evaluate individuals with stale fitness
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        # Reassemble population: elite + offspring
        population[:] = elite + offspring

        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, n_evals=len(invalid), **record)

        # Print only when a new global best is found
        new_best = hof[0].fitness.values[0]
        if verbose and new_best < prev_best:
            _print_generation(gen, record, hof[0])
            prev_best = new_best

    return population, logbook, hof[0]


def _print_generation(
    gen: int,
    record: dict,
    best_ind: creator.Individual,
) -> None:
    """Print a one-line per-generation summary."""
    best_power  = best_ind.fitness.values[0]   # penalised power (W)
    best_params = genome_to_dict(best_ind)
    from physics import compute_wheelbase, calculate_cda, RIDER_MASS
    from constraints import check_uci_compliance
    wb    = compute_wheelbase(best_params)
    cr    = check_uci_compliance(best_params)
    legal = 'UCI-OK' if cr.is_legal else 'ILLEGAL'
    cda   = calculate_cda(best_params)
    f_wd  = best_params['front_wheel_diameter']
    r_wd  = best_params['rear_wheel_diameter']
    wheel_diff = abs(f_wd - r_wd)
    wheel_flag = (f"  ★ ASYMMETRIC WHEELS: {f_wd:.0f}F/{r_wd:.0f}R (+{wheel_diff:.0f}mm)"
                  if wheel_diff > 50 else f"  wheels:{f_wd:.0f}F/{r_wd:.0f}R")
    print(
        f"Gen {gen:>4d} | "
        f"Best P={best_power:.1f} W | "
        f"Avg={record['mean']:.1f} W | "
        f"CdA={cda:.4f} m² | "
        f"WB={wb:.0f} mm | "
        f"{legal}"
        f"{wheel_flag}"
    )
