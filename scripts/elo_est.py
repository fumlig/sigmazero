#!/usr/bin/env python3

"""Estimate Elo rating of UCI engine."""


import argparse
import multiprocessing as mp
import time
import os
import chess
import chess.engine


def elo(player_rating: float, opponent_rating: float, score: float, k: int) -> float:
	"""Elo rating for player given old rating, opponent's rating and actual score."""
	expected = 1 / (1 + 10**((opponent_rating - player_rating)/400))
	return player_rating + k*(score - expected)


def fide_rating(player_rating: float, opponent_rating: float, score: float):
	"""FIDE uses an approximation to the logistic distribution for."""
	if player_rating < 2300: 	k = 40
	elif player_rating < 2400: 	k = 20
	else:						k = 10
	return elo(player_rating, opponent_rating, score, k)


def play(engine_path: str, judge_path: str, time: float, rating: mp.Value, outcomes: mp.Queue, done: mp.Event):
	"""Play games between engine and judge."""
	engine = chess.engine.SimpleEngine.popen_uci(engine_path)
	judge = chess.engine.SimpleEngine.popen_uci(judge_path)
	board = chess.Board()
	side = chess.WHITE
	limit = chess.engine.Limit(time=time)
	judge_min = judge.options["UCI_Elo"].min
	judge_max = judge.options["UCI_Elo"].max

	while not done.is_set():

		# match ratings
		judge_rating = max(rating.value, judge_min)

		if judge_rating > judge_max:
			judge.configure({"UCI_LimitStrength": False})
		else:
			judge.configure({"UCI_LimitStrength": True})
			judge.configure({"UCI_Elo": judge_rating})

		while not board.is_game_over():

			if board.turn == side:
				r = engine.play(board, limit)
			else:
				r = judge.play(board, limit)
			
			board.push(r.move)

		outcome = board.outcome()

		if outcome.winner == side:		score = 1.0
		elif outcome.winner == None:	score = 0.5
		else:							score = 0.0

		outcomes.put((score, judge_rating))
		side = not side
		board.reset()

	engine.close()
	judge.close()


def estimate(engine_path: str, judge_path: str = "/usr/bin/stockfish", workers: int = os.cpu_count(), time: float = 0.01, window_size: int = 30, threshold: float = 5, guess: float = 1500):
	"""
	Estimate Elo rating of engine.
	
	engine_path:	path to engine executable
	judge_path:		path to judge executable (strength should be tunable via UCI)
	workers:		number of processes to use for playing games
	time:			time allowed per move
	window_size:	number of most recent games to consider for convergence check
	threshold:		stop when mean absolute error in window is below this value
	guess:			initial engine rating guess

	return:			rating and mean absolute error after each played game until convergence
	"""
	with mp.Manager() as manager:
		result = guess
		history = [result]
		rating = manager.Value('d', result)
		outcomes = manager.Queue()
		done = manager.Event()

		workers = [mp.Process(target=play, args=(engine_path, judge_path, time, rating, outcomes, done)) for _ in range(workers)]

		for worker in workers:
			worker.start()

		while not done.is_set():
			score, judge_rating = outcomes.get()
			result = fide_rating(rating.value, judge_rating, score)
			history.append(result)
			rating.value = result

			window = history[-window_size:]
			mean = sum(window)/len(window)
			mae = sum(abs(mean - r) for r in window)/len(window)

			yield result, mae

			if len(history) >= window_size and mae <= threshold:
				#result = mean
				done.set()
		
		for worker in workers:
			worker.join()
		
		return result, mae



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Estimate Elo rating of UCI engine.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("engine", type=str, help="Path to engine executable")
	parser.add_argument("--judge", type=str, default="/usr/bin/stockfish", help="Path to judge executable")
	parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Processes to use")
	parser.add_argument("--time", type=float, default=0.01, help="Time per move")
	parser.add_argument("--window", type=int, default=30, help="Window of ratings to use for stopping condition (larger window means more certainty)")
	parser.add_argument("--threshold", type=int, default=5, help="Stop when mean absolute error of ratings in window is below this value (lower threshold means more certainty)")
	parser.add_argument("--guess", type=float, default=1500, help="A good rating guess can speed up evaluation considerably.")

	args = parser.parse_args()

	begin = time.time()
	
	for rating, mae in estimate(args.engine, args.judge, args.workers, args.time, args.window, args.threshold, args.guess):
		print(f"rating: {rating:.2f}, mae: {mae:.2f}")
	
	end = time.time()
