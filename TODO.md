
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



# To do...

* CHECK: Make sure build successful, lobby dockerfile published.  Set up on server
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
