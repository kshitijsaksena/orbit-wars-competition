import math
import random
from collections import namedtuple

# Define local namedtuples for convenience and robustness
Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

# Game Constants
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def point_to_segment_distance(p, v, w):
    """Minimum distance from point p to line segment v-w."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return distance(p, v)
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)

def swept_pair_hit(A, B, P0, P1, r):
    """True iff a fleet moving A->B and a planet moving P0->P1 come within r of each other."""
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0

def get_fleet_speed(ships, max_speed=6.0):
    if ships <= 1:
        return 1.0
    val = (math.log(ships) / math.log(1000.0))
    if val < 0.0:
        val = 0.0
    speed = 1.0 + (max_speed - 1.0) * (val ** 1.5)
    return min(speed, max_speed)

_position_cache = {}
_planet_params_cache = {}
_cache_metadata = {"angular_velocity": None, "step": -1}

def get_planet_position_at_tick(planet_id, tick, obs):
    """Returns (x, y) coordinates of a planet or comet at a specific tick, using caches for optimization."""
    ang_vel = obs.get("angular_velocity", 0.0)
    step = obs.get("step", 0)
    
    if _cache_metadata["angular_velocity"] != ang_vel or step < _cache_metadata["step"]:
        _position_cache.clear()
        _planet_params_cache.clear()
        _cache_metadata["angular_velocity"] = ang_vel
        
    _cache_metadata["step"] = step
    
    key = (planet_id, tick)
    if key in _position_cache:
        return _position_cache[key]
        
    comet_ids = obs.get("comet_planet_ids", [])
    if planet_id in comet_ids:
        # Comet position from paths
        for group in obs.get("comets", []):
            if planet_id in group["planet_ids"]:
                idx = group["planet_ids"].index(planet_id)
                t_diff = tick - step
                future_idx = group["path_index"] + t_diff
                path = group["paths"][idx]
                if 0 <= future_idx < len(path):
                    pos = (path[future_idx][0], path[future_idx][1])
                    _position_cache[key] = pos
                    return pos
                else:
                    _position_cache[key] = None
                    return None
        _position_cache[key] = None
        return None

    # Regular planet position
    if planet_id not in _planet_params_cache:
        initial_planets = obs.get("initial_planets", [])
        planet = next((p for p in initial_planets if p[0] == planet_id), None)
        if planet is None:
            _position_cache[key] = None
            return None
        dx = planet[2] - CENTER
        dy = planet[3] - CENTER
        r = math.sqrt(dx**2 + dy**2)
        is_orbiting = (r + planet[4] < ROTATION_RADIUS_LIMIT)
        _planet_params_cache[planet_id] = (dx, dy, r, is_orbiting, planet[2], planet[3])
        
    dx, dy, r, is_orbiting, px, py = _planet_params_cache[planet_id]
    if is_orbiting:
        initial_angle = math.atan2(dy, dx)
        current_angle = initial_angle + ang_vel * (tick - 1) if tick >= 1 else initial_angle
        pos = (
            CENTER + r * math.cos(current_angle),
            CENTER + r * math.sin(current_angle)
        )
    else:
        pos = (px, py)
        
    _position_cache[key] = pos
    return pos

def get_comet_expiry_step(comet_id, obs):
    for group in obs.get("comets", []):
        if comet_id in group["planet_ids"]:
            current_step = obs.get("step", 0)
            idx = group["planet_ids"].index(comet_id)
            path = group["paths"][idx]
            remaining_steps = len(path) - group["path_index"]
            return current_step + remaining_steps
    return 500

def simulate_trajectory(source_id, target_id, start_pos, angle, speed, obs):
    """Simulates fleet trajectory and returns (success, arrival_step)."""
    T = obs.get("step", 0)
    planets = obs.get("planets", [])
    source_radius = next(p[4] for p in planets if p[0] == source_id)
    offset = source_radius + 0.1
    F_start = (start_pos[0] + math.cos(angle) * offset, start_pos[1] + math.sin(angle) * offset)
    
    max_steps = 150
    for s in range(T, T + max_steps):
        old_f = (F_start[0] + (s - T) * speed * math.cos(angle), F_start[1] + (s - T) * speed * math.sin(angle))
        new_f = (F_start[0] + (s - T + 1) * speed * math.cos(angle), F_start[1] + (s - T + 1) * speed * math.sin(angle))
        
        # Out of bounds check
        if not (0 <= new_f[0] <= BOARD_SIZE and 0 <= new_f[1] <= BOARD_SIZE):
            return False, None
            
        # Sun collision check (with safety buffer)
        if point_to_segment_distance((CENTER, CENTER), old_f, new_f) < (SUN_RADIUS + 0.2):
            return False, None
            
        # Check planet collisions
        hit_target = False
        hit_other = False
        for p in planets:
            pid = p[0]
            radius = p[4]
            p_old = get_planet_position_at_tick(pid, max(0, s - 1), obs)
            p_new = get_planet_position_at_tick(pid, s, obs)
            if p_old is None or p_new is None:
                continue
                
            if swept_pair_hit(old_f, new_f, p_old, p_new, radius):
                if pid == target_id:
                    hit_target = True
                elif pid != source_id:
                    hit_other = True
                    
        if hit_other:
            return False, None
        if hit_target:
            return True, s
            
    return False, None

def get_intercept_angle_and_time(source_planet, target_planet, speed, obs):
    """Calculates intercept angle and arrival time for a fleet launch."""
    T = obs.get("step", 0)
    S_start = (source_planet.x, source_planet.y)
    T_curr = (target_planet.x, target_planet.y)
    D = distance(S_start, T_curr)
    offset = source_planet.radius + 0.1
    estimated_K = int(round((D - offset) / speed))
    if estimated_K < 1:
        estimated_K = 1
        
    best_angle = None
    best_arrival = None
    min_arrival = float("inf")
    
    # Check estimated K and narrow neighbors for speed optimization, but expand for comets
    comet_ids = obs.get("comet_planet_ids", [])
    if target_planet.id in comet_ids:
        candidate_Ks = list(range(max(1, estimated_K - 10), estimated_K + 25))
    else:
        candidate_Ks = [estimated_K]
        if estimated_K > 1:
            candidate_Ks.append(estimated_K - 1)
        candidate_Ks.append(estimated_K + 1)
    
    for K in candidate_Ks:
        t_arr = T + K
        P_target = get_planet_position_at_tick(target_planet.id, t_arr, obs)
        if P_target is None:
            continue
            
        angle = math.atan2(P_target[1] - S_start[1], P_target[0] - S_start[0])
        
        # Test angle and small variations
        for angle_offset in [0.0, -0.02, 0.02, -0.05, 0.05]:
            candidate_angle = angle + angle_offset
            success, arrival_step = simulate_trajectory(source_planet.id, target_planet.id, S_start, candidate_angle, speed, obs)
            if success:
                if arrival_step < min_arrival:
                    min_arrival = arrival_step
                    best_angle = candidate_angle
                    best_arrival = arrival_step
                    break
                    
    if best_angle is not None:
        return best_angle, best_arrival
        
    # Fallback to direct angle
    angle = math.atan2(target_planet.y - S_start[1], target_planet.x - S_start[0])
    success, arrival_step = simulate_trajectory(source_planet.id, target_planet.id, S_start, angle, speed, obs)
    if success:
        return angle, arrival_step
        
    return None, None

_fleet_dest_cache = {}

def predict_fleet_destination(fleet, planets, obs):
    """Identifies destination planet and arrival step of a fleet in flight, caching results."""
    f_id = fleet[0]
    if f_id in _fleet_dest_cache:
        return _fleet_dest_cache[f_id]
        
    f_owner, fx, fy, f_angle, from_pid, f_ships = fleet[1:]
    speed = get_fleet_speed(f_ships)
    T = obs.get("step", 0)
    
    for s in range(T, T + 150):
        old_f = (fx + (s - T) * speed * math.cos(f_angle), fy + (s - T) * speed * math.sin(f_angle))
        new_f = (fx + (s - T + 1) * speed * math.cos(f_angle), fy + (s - T + 1) * speed * math.sin(f_angle))
        
        if not (0 <= new_f[0] <= BOARD_SIZE and 0 <= new_f[1] <= BOARD_SIZE):
            _fleet_dest_cache[f_id] = (None, None)
            return None, None
        if point_to_segment_distance((CENTER, CENTER), old_f, new_f) < SUN_RADIUS:
            _fleet_dest_cache[f_id] = (None, None)
            return None, None
            
        for p in planets:
            pid = p.id
            radius = p.radius
            p_old = get_planet_position_at_tick(pid, max(0, s - 1), obs)
            p_new = get_planet_position_at_tick(pid, s, obs)
            if p_old is None or p_new is None:
                continue
            if swept_pair_hit(old_f, new_f, p_old, p_new, radius):
                _fleet_dest_cache[f_id] = (pid, s)
                return pid, s
    _fleet_dest_cache[f_id] = (None, None)
    return None, None

def simulate_planet_state_to_step(planet, target_step, incoming_fleets, obs):
    """Simulates planet state at target_step, considering production and incoming fleets."""
    current_step = obs.get("step", 0)
    owner = planet.owner
    ships = planet.ships
    prod = planet.production
    
    fleets_by_step = {}
    for f_owner, f_ships, arr_step in incoming_fleets:
        if arr_step not in fleets_by_step:
            fleets_by_step[arr_step] = []
        fleets_by_step[arr_step].append((f_owner, f_ships))
        
    for s in range(current_step + 1, target_step + 1):
        if owner != -1:
            ships += prod
            
        if s in fleets_by_step:
            player_ships = {}
            for f_owner, f_ships in fleets_by_step[s]:
                player_ships[f_owner] = player_ships.get(f_owner, 0) + f_ships
                
            if player_ships:
                sorted_players = sorted(player_ships.items(), key=lambda x: x[1], reverse=True)
                top_player, top_ships = sorted_players[0]
                
                if len(sorted_players) > 1:
                    second_ships = sorted_players[1][1]
                    survivor_ships = top_ships - second_ships
                    if sorted_players[0][1] == sorted_players[1][1]:
                        survivor_ships = 0
                    survivor_owner = top_player if survivor_ships > 0 else -1
                else:
                    survivor_owner = top_player
                    survivor_ships = top_ships
                    
                if survivor_ships > 0:
                    if owner == survivor_owner:
                        ships += survivor_ships
                    else:
                        ships -= survivor_ships
                        if ships < 0:
                            owner = survivor_owner
                            ships = abs(ships)
                            
    return owner, ships

def agent(obs, config=None):
    player = obs.get("player", 0)
    current_step = obs.get("step", 0)
    
    if current_step == 0:
        _fleet_dest_cache.clear()
    
    # Parse planets and fleets
    planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets_raw = obs.get("fleets", [])
    
    # Build list of active fleets
    fleets = [Fleet(*f) for f in fleets_raw]
    
    # Pre-calculate incoming fleets for each planet
    incoming_by_planet = {p.id: [] for p in planets}
    for f in fleets:
        dest_id, arr_step = predict_fleet_destination(f, planets, obs)
        if dest_id is not None and dest_id in incoming_by_planet:
            incoming_by_planet[dest_id].append((f.owner, f.ships, arr_step))
            
    # Track available ships per planet
    planet_avail_ships = {p.id: p.ships for p in planets}
    
    moves = []
    
    # 1. Defend threatened home planets
    threatened_planets = []
    for p in planets:
        if p.owner == player:
            # Simulate planet state at the end of the game or next 50 steps
            sim_steps = min(500, current_step + 40)
            end_owner, end_ships = simulate_planet_state_to_step(p, sim_steps, incoming_by_planet[p.id], obs)
            if end_owner != player:
                # Find when the loss happens
                # Get the first step where ownership is lost
                for s in range(current_step + 1, sim_steps + 1):
                    s_owner, s_ships = simulate_planet_state_to_step(p, s, incoming_by_planet[p.id], obs)
                    if s_owner != player:
                        threatened_planets.append((p.id, s, s_ships))
                        break
                        
    # Sort threatened planets by urgency (earlier step first)
    threatened_planets.sort(key=lambda x: x[1])
    
    for tp_id, loss_step, needed_ships in threatened_planets:
        tp = next(p for p in planets if p.id == tp_id)
        # Find friendly helper planets
        helpers = [p for p in planets if p.owner == player and p.id != tp_id]
        
        # Sort helpers by distance to target
        helpers.sort(key=lambda hp: distance((hp.x, hp.y), (tp.x, tp.y)))
        
        ships_still_needed = needed_ships + 1
        for hp in helpers:
            if ships_still_needed <= 0:
                break
                
            hp_avail = planet_avail_ships[hp.id]
            # Keep a small reserve on the helper
            reserve = max(5, int(hp.ships * 0.15))
            sendable = hp_avail - reserve
            if sendable <= 0:
                continue
                
            # Estimate speed and check if we can arrive in time
            fleet_size = min(sendable, ships_still_needed)
            speed = get_fleet_speed(fleet_size)
            angle, arr_step = get_intercept_angle_and_time(hp, tp, speed, obs)
            
            if arr_step is not None and arr_step < loss_step:
                # We can reinforce in time!
                moves.append([hp.id, angle, fleet_size])
                planet_avail_ships[hp.id] -= fleet_size
                # Add to simulation so other helpers know we sent reinforcement
                incoming_by_planet[tp.id].append((player, fleet_size, arr_step))
                ships_still_needed -= fleet_size
                
    # 2. Offensive Target Selection & Expansion
    candidates = []
    my_planets = [p for p in planets if p.owner == player]
    
    for mp in my_planets:
        hp_avail = planet_avail_ships[mp.id]
        reserve = max(5, int(mp.ships * 0.15))
        sendable = hp_avail - reserve
        if sendable <= 0:
            continue
            
        for tp in planets:
            if tp.owner == player:
                continue
                
            # Cheap transit estimate using Euclidean distance
            speed = get_fleet_speed(sendable)
            offset = mp.radius + 0.1
            dist = distance((mp.x, mp.y), (tp.x, tp.y))
            est_transit = int(round((dist - offset) / speed))
            if est_transit < 1:
                est_transit = 1
                
            est_arr_step = current_step + est_transit
            
            # Check comet expiration for estimate
            comet_ids = obs.get("comet_planet_ids", [])
            expiry = 500
            if tp.id in comet_ids:
                expiry = get_comet_expiry_step(tp.id, obs)
                if est_arr_step >= expiry:
                    continue
                    
            # Estimate target ships at transit arrival step
            est_owner, est_ships = simulate_planet_state_to_step(tp, est_arr_step, incoming_by_planet[tp.id], obs)
            if est_owner == player:
                continue
                
            est_required = est_ships + 1
            if sendable < est_required:
                # Impossible to conquer with current sendable ships, skip exact trajectory search!
                continue
                
            # Now we do the heavy exact intercept simulation
            angle, arr_step = get_intercept_angle_and_time(mp, tp, speed, obs)
            if arr_step is None:
                continue
                
            if tp.id in comet_ids and arr_step >= expiry:
                continue
                
            # Re-simulate with exact arrival step
            sim_owner, sim_ships = simulate_planet_state_to_step(tp, arr_step, incoming_by_planet[tp.id], obs)
            if sim_owner == player:
                continue
                
            required = sim_ships + 1
            if sendable >= required:
                # Calculate actual fleet size and re-verify intercept
                fleet_size = required
                actual_speed = get_fleet_speed(fleet_size)
                actual_angle, actual_arr_step = get_intercept_angle_and_time(mp, tp, actual_speed, obs)
                
                if actual_arr_step is not None:
                    if tp.id in comet_ids and actual_arr_step >= expiry:
                        continue
                        
                    transit = max(1, actual_arr_step - current_step)
                    lifetime = expiry - actual_arr_step
                    
                    # ROI score
                    score = (tp.production * lifetime) / (max(1, fleet_size) * (transit ** 1.25))
                    candidates.append((score, mp.id, tp.id, actual_angle, fleet_size, actual_arr_step))
                    
    # Sort candidates by ROI score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    for score, mp_id, tp_id, angle, fleet_size, arr_step in candidates:
        # Verify source planet still has enough available ships
        if planet_avail_ships[mp_id] - fleet_size >= max(5, int(next(p.ships for p in planets if p.id == mp_id) * 0.15)):
            moves.append([mp_id, angle, fleet_size])
            planet_avail_ships[mp_id] -= fleet_size
            incoming_by_planet[tp_id].append((player, fleet_size, arr_step))
            
    # 3. Endgame all-in (Turns > 480)
    # Launch remaining ships to closest targets to maximize final score
    if current_step > 480:
        for mp in my_planets:
            hp_avail = planet_avail_ships[mp.id]
            if hp_avail > 0:
                # Find closest enemy or neutral planet
                targets = [p for p in planets if p.owner != player]
                if targets:
                    targets.sort(key=lambda t: distance((mp.x, mp.y), (t.x, t.y)))
                    closest = targets[0]
                    speed = get_fleet_speed(hp_avail)
                    angle, arr_step = get_intercept_angle_and_time(mp, closest, speed, obs)
                    if angle is not None:
                        moves.append([mp.id, angle, hp_avail])
                        planet_avail_ships[mp.id] = 0
                        
    return moves
