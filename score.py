from dataclasses import dataclass

from board import Board, build_board
from snap import Placement, load_pipeline, read_photo

PLAYERS = ["red", "blue", "white", "orange"]
LONGEST_ROAD_MIN = 5
LONGEST_ROAD_POINTS = 2
BUILDING_POINTS = {"settlement": 1, "city": 2}


@dataclass
class PlayerScore:
    color: str
    settlements: int
    cities: int
    roads: int
    longest_road: int
    has_longest_road: bool
    points: int


def building_owners(placements: list[Placement]) -> dict[int, str]:
    return {p.target: p.color for p in placements if p.kind == "vertex" and p.color is not None}


def road_graph(board: Board, placements: list[Placement], color: str) -> dict[int, list[tuple[int, int]]]:
    graph = {}
    for p in placements:
        if p.piece == "road" and p.color == color:
            a, b = board.edges[p.target]
            if a not in graph:
                graph[a] = []
            if b not in graph:
                graph[b] = []
            graph[a].append((p.target, b))
            graph[b].append((p.target, a))
    return graph


def longest_trail(vertex: int, graph: dict[int, list[tuple[int, int]]], used: set[int],
                  owners: dict[int, str], color: str) -> int:
    best = 0
    if vertex not in graph:
        return 0
    for edge_id, next_vertex in graph[vertex]:
        if edge_id in used:
            continue
        used.add(edge_id)
        owner = owners.get(next_vertex)
        if owner is not None and owner != color:
            length = 1
        else:
            length = 1 + longest_trail(next_vertex, graph, used, owners, color)
        used.remove(edge_id)
        
        best = max(length, best)
    return best


def longest_road(board: Board, placements: list[Placement], color: str) -> int:
    graph = road_graph(board, placements, color)
    owners = building_owners(placements)
    if not graph:
        return 0
    largest = 0
    for v in graph:
        curr_size = (longest_trail(v, graph, set(), owners, color))
        largest = max(curr_size, largest)
    return largest
        


def award_longest_road(lengths: dict[str, int]) -> str | None:
    longest_length = 0
    longest_color = []
    for color, length in lengths.items():
        if length > longest_length:
            if longest_color: 
                while len(longest_color) != 0:
                    longest_color.pop()
            longest_length = length
            longest_color.append(color)
        elif length == longest_length:
            longest_color.append(color)
    if len(longest_color) > 1 or longest_length < LONGEST_ROAD_MIN:
        return None
    return longest_color[0]
            
        


def score(board: Board, placements: list[Placement]) -> list[PlayerScore]:
    players = [color for color in PLAYERS if any(p.color == color for p in placements)]
    lengths = {color: longest_road(board, placements, color) for color in players}
    winner = award_longest_road(lengths)

    scores = []
    for color in players:
        mine = [p for p in placements if p.color == color]
        counts = {piece: sum(p.piece == piece for p in mine) for piece in ("settlement", "city", "road")}
        points = sum(BUILDING_POINTS[piece] * counts[piece] for piece in BUILDING_POINTS)
        if color == winner:
            points += LONGEST_ROAD_POINTS
        scores.append(PlayerScore(color, counts["settlement"], counts["city"], counts["road"],
                                  lengths[color], color == winner, points))
    return sorted(scores, key=lambda s: s.points, reverse=True)


def main() -> None:
    loaded = load_pipeline()
    if loaded is None:
        return
    model, color_model, corners = loaded
    board = build_board()

    for pid, clicked in corners.items():
        placements = read_photo(pid, clicked, model, color_model, board).placements
        scores = score(board, placements)
        if not scores:
            print(f"{pid}: no pieces on the board")
            continue
        print(pid)
        for s in scores:
            badge = "  + Longest Road" if s.has_longest_road else ""
            print(f"  {s.color:7} {s.points:2} pts   {s.settlements} settlements, {s.cities} cities, "
                  f"{s.roads} roads, longest {s.longest_road}{badge}")


if __name__ == "__main__":
    main()
