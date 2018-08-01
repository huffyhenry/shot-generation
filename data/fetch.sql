SELECT
  g.kickoff::DATE AS date,
  t1.name AS team1,
  t2.name AS team2,
  round(e.min + e.sec/60.0, 2) AS time,
  round(e.min + e.sec/60.0 - lag(e.min + e.sec/60.0, 1, 0.0) OVER half, 2) AS wait,
  t.name AS team,
  CASE WHEN t.id = t1.id THEN t2.name ELSE t1.name END AS oppo,
  (e.team_id = g.team1_id)::INT AS home,
  0 AS neutral,
  (e.type_id != 30)::INT AS shot,
  (e.type_id = 16)::INT AS goal,
  (q_own.id IS NOT NULL)::INT AS own_goal,
  (q_pen.id IS NOT NULL)::INT AS penalty,
  game_state AS state
FROM
  base_event e
  JOIN base_team t ON e.team_id = t.id AND e.type_id IN (13, 14, 15, 16, 30)
  JOIN base_game g on e.game_id = g.id
  JOIN base_team t1 on g.team1_id = t1.id
  JOIN base_team t2 on g.team2_id = t2.id
  JOIN base_tournament c on g.tournament_id = c.id
  LEFT JOIN base_qualifier q_own ON e.id = q_own.event_id AND q_own.type_id = 28
  LEFT JOIN base_qualifier q_pen ON e.id = q_pen.event_id AND q_pen.type_id = 9
WHERE
  e.period_id IN (1,2)  -- The End event (type_id=30) also exists for period_id=14
  AND (e.type_id != 30 OR e.team_id = g.team1_id)  -- Pick one End event of the pair
  AND c.region = 'England'
  AND c.competition = 'Premier League'
  AND g.kickoff BETWEEN '2017-07-01' AND '2018-07-01'
WINDOW half AS (PARTITION BY e.game_id, e.period_id ORDER BY e.min, e.sec)
ORDER BY
  date, g.id, time

