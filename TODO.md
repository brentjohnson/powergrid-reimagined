
# v0.3.0

* BUG: Said Nick won. Brand and Nick powered 13, but Brad had more money.
* BUG: I owned PP 31, but it wouldn't let me buy coal?  It showed 0/6?
  * NOTE: I got rid of an Oil plant and bought a coal.  The new coal is what wasn't working (even though it showed)
  * NOTE: The cause of this was the invalid resource after losing a plant (two bugs, one prompt)
* IMPROVEMENT: Not obvious which plant is up for bid
* IMPROVEMENT: Bidding should happen in action panel.
* BUG: When losing a powerplant, check for invalid resources
* BUG: Step 2 started too soon (during city building)
* BUG: Step 3.  Is the card in there?  Does it work?  Will power plant market adjust?
* BUG: Limit the number of regions
* BUG: Choose which plant to discard
* BUG: Step 3, power plant market not reloaded (cards not put on bottom of deck?)
* BUG: Owned 20,29,42. 15 cities. 5 coal, 2 oil.  It only powered 11 cities???
* BUG: My bid was excessively high... did it remember my old bid or something?
* BUG: Lowest power plant is discarded after each auction round.
* CHECK: Does it discard the 3 when third city built?
* IMPROVEMENT: Make connection string easier (just the host, not the rest - default to onyxoryx)

# v0.3.1

* IMPROVEMENT: Computer controlled player!
* BUG: Player should be able to choose resources for powering (hybrid!)
* BUG: Player should be able to choose resource to lose when discarding plant
* TECH DEBT: Optimize selecting discard resources. Seems like lots of code for the two times its used.
* CHECK: Are all cities properly assigned to regions?
* BUG: Number of end game cities is not right.
* IMPROVEMENT: Exit game besides ALT-F4 (ESC menu: what other menu things?)
* BUG: Client just closes at end of game. No time to review results.

# v0.3.2

* TECH DEBT: Upgrade to Bevy 0.17
* TECH DEBT: Upgrade to Bevy 0.18

# v0.3.3

* BUG: Fix container build

# v0.3.4

* IMPROVEMENT: Auction mode... List each player in a column and show status/bids.
* IMPROVEMENT: Smarter bots

# v0.3.5

* IMPROVEMENT: House shape rather than circles for cities.
* IMPROVEMENT: Smarter bots (don't buy so many resources)
* IMPROVEMENT: Show replenishment on resource market

# v0.4.0

* Add lobby to launch games

# v0.4.1

* Fix build

# v0.5.0

* Local (offline) play

# v0.5.1

* IMPROVEMENT: implement user accounts


# To do...

* BUG: local game not showing individual bot actions
* BUG: local game "back to main menu" crashes client
* IMPROVEMENT: Remove connect screen for on-line. Just use default or command line arg
* IMPROVEMENT: Dynamically refresh active game list. Remove "refresh" button
* IMPROVEMENT: Click on "active" game to populate the name
* IMPROVEMENT: Existing games should be marked started/not started and if you are in them
* IMPROVEMENT: don't select color on connect, select for each game
* IMPROVEMENT: user can save color preferences in account (order of pref)
* IMPROVEMENT: implement user statistics
* IMPROVEMENT: implement game statistics
* IMPROVEMENT: implement game state save
* IMPROVEMENT: configurable bot speed (in settings menu)
* IMPROVEMENT: Square rather than circle hit-shape for city.  Add city name. Cover connection lines.
* BUG: Two people pick same color, can't start game
* COMPLAINT: Less reading for Nick
* IMPROVEMENT: Randomize initial order
* IMPROVEMENT: Do not reorder players on left (that should remain bidding order) DEBATABLE!?
* IMPROVEMENT: Ding on your turn
* IMPROVEMENT: Add "capacity" to player card (how many you could power)
* IMPROVEMENT: Show remaining money when doing resources or cities?
* IMPROVEMENT: Implement dependabot on the repo
* DEBUG: log more info
* DEBUG: dump state for diagnostics
* DEBUG: view hidden info (deck)
* FUTURE: Random maps? or use Google maps? 
* FUTURE: AI training of bots (pettingzoo?)

 ┌────────────────────┬────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
 │        Data        │                     Storage shape                      │                                 Why Postgres fits                                 │
 ├────────────────────┼────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ User accounts      │ Normal table: id, email, username, password_hash, …    │ Inherently relational — login lookups by unique columns, FK targets. Needs ACID.  │
 ├────────────────────┼────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ User statistics    │ Tables keyed by user_id (games_played, wins, elo…)     │ Aggregations, joins to accounts, indexed leaderboards.                            │
 ├────────────────────┼────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ Game statistics    │ games table + game_players join + optional             │ Relational queries ("avg score by plant id") with JSONB escape hatch for          │
 │                    │ events_jsonb for raw history                           │ unstructured detail.                                                              │
 ├────────────────────┼────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ In-progress game   │ One row per room: room_id PRIMARY KEY, state JSONB,    │ Turn-based mutation rate is trivially handled by Postgres. JSONB is indexable and │
 │ state              │ updated_at                                             │  avoids a second datastore.                                                       │
 └────────────────────┴────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘
