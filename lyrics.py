

import urllib
import sys
import os
import re
import subprocess
import lxml.html
import urllib.request

def lyricwikicase(s):
	"""Return a string in LyricWiki case.
	Substitutions are performed as described at 
	<http://lyrics.wikia.com/LyricWiki:Page_Names>.
	Essentially that means capitalizing every word and substituting certain 
	characters."""

	words = s.split()
	newwords = []
	for word in words:
		newwords.append(word[0].capitalize() + word[1:])
	s = "_".join(newwords)
	s = s.replace("<", "Less_Than")
	s = s.replace(">", "Greater_Than")
	s = s.replace("#", "Number_") # FIXME: "Sharp" is also an allowed substitution
	s = s.replace("[", "(")
	s = s.replace("]", ")")
	s = s.replace("{", "(")
	s = s.replace("}", ")")
	try:
		# Python 3 version
		s = urllib.parse.urlencode([(0, s)])[2:]
	except AttributeError:
		# Python 2 version
		s = urllib.urlencode([(0, s)])[2:]
	return s

def lyricwikipagename(artist, title):
	"""Return the page name for a set of lyrics given the artist and 
	title"""

	return "%s:%s" % (lyricwikicase(artist), lyricwikicase(title))

def lyricwikiurl(artist, title, edit=False, fuzzy=False):
	"""Return the URL of a LyricWiki page for the given song, or its edit 
	page"""

	if fuzzy:
		base = "http://lyrics.wikia.com/index.php?search="
		pagename = artist + ':' + title
		if not edit:
			url = base + pagename
			doc = lxml.html.parse(url)
			return doc.docinfo.URL
	else:
		base = "http://lyrics.wikia.com/"
		pagename = lyricwikipagename(artist, title)
	if edit:
		if fuzzy:
			url = base + pagename
			doc = lxml.html.parse(url)
			return doc.docinfo.URL + "&action=edit"
		else:
			return base + "index.php?title=%s&action=edit" % pagename
	return base + pagename

def __executableexists(program):
	"""Determine whether an executable exists"""

	for path in os.environ["PATH"].split(os.pathsep):
		exefile = os.path.join(path, program)
		if os.path.exists(exefile) and os.access(exefile, os.X_OK):
			return True
	return False

def currentlyplaying():
	"""Return a tuple (artist, title) if there is a currently playing song in 
	MPD, otherwise None.
	Raise an OSError if no means to get the currently playing song exist."""

	artist = None
	title = None

	if not __executableexists("mpc"):
		raise OSError("mpc is not available")

	output = subprocess.Popen(["mpc", "--format", "%artist%\\n%title%"],
			stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
	if not output.startswith("volume: "):
		(artist, title) = output.splitlines()[0:2]

	if artist is None or title is None:
		return None
	return (artist, title)

def getlyrics(artist, title, fuzzy=False):
	"""Get and return the lyrics for the given song.
	Raises an IOError if the lyrics couldn't be found.
	Raises an IndexError if there is no lyrics tag.
	Returns False if there are no lyrics (it's instrumental)."""

	try:
		url = lyricwikiurl(artist, title, fuzzy=fuzzy)

		doc = lxml.html.parse(urllib.request.urlopen(url))
	except IOError:
		raise

	try:
		lyricbox = doc.getroot().cssselect(".lyricbox")[0]
	except IndexError:
		raise

	# look for a sign that it's instrumental
	if len(doc.getroot().cssselect(".lyricbox a[title=\"Instrumental\"]")):
		return False

	# prepare output
	lyrics = []
	if lyricbox.text is not None:
		lyrics.append(lyricbox.text)
	for node in lyricbox:
		if str(node.tag).lower() == "br":
			lyrics.append("\n")
		if node.tail is not None:
			lyrics.append(node.tail)
	return "".join(lyrics).strip()
