#to diavasma ths vasis dedomenwn kai to kopsimo twn papers ginetai edw
from timeit import default_timer as timer
import random
import pickle
import sys
from sortedcontainers import SortedDict
from Paper import *

DEBUG = 0
def dprint(msg):
	if DEBUG == 1:
		print("'"+str(msg)+"'")

def check_continuity(paper):
	timeline = paper.timeline_of_citations
	cont = 1
	prev_year = next(iter(timeline))-1
	for year in timeline:
		if year != prev_year + 1:
			return 0
	return 1

def searchPaper(papers,id):
	for paper in papers:
		print(paper.indexID,id)
		if paper.indexID == id:
			return paper                        
	
def main():
	paperinfo = []
	file_lines = []
	papers = []
	linesRead = 0
	
	
	# if len(sys.argv) > 1 and sys.argv[1] == '--skip-big':
	start = timer()
	with open("outputacm.txt", "r") as file:
		lines = []
		for line in file:
			file_lines.append(line)
	end = timer()
	elapsed = end - start
	print("Reading 'outputacm.txt' in "+str(elapsed) +" seconds")

	start = timer()
	for line in file_lines:
		#paper info ended.Put all information to a Paper class, cause in next loop new info starts
		if line[0] == '\n':
			paperinfo.append(lines)
			dprint("paperinfo : ")
			dprint(paperinfo)
			paper = Paper()
			paper.parse(paperinfo)
			papers.append(paper)
			lines=[]
			paperinfo = []
		else:
			lines.append(line)
		#	return
		# if(linesRead > 0):
		#     line = line.strip()
		#     lines.append(line)
		#     print("read line: "+line)
		
		linesRead += 1

			
		#print("paperinfo : %s", paperinfo)
	print("Finished making papers")
	end = timer()
	elapsed = end-start

	# with open('test_pickle','wb') as handle:
		#pickle.dump(papers,handle)

	print("Making papers in "+str(elapsed) +" seconds")
	

	start = timer()
	analyzing_years = []
	
	for paper in papers:
		year = paper.Year
		refs = paper.referenceIDs
		#print(refs)
		for ref in refs:
			papers[ref].incrementTimelineOfCitations(year)
		
	print("How much papers: "+str(len(papers)))


	if len(sys.argv) != 3:
		print('Give me 2 arguments:timelength start, timelength end, so that i can extract from start to end-1')
		sys.exit()
	timeline_length_start = int(sys.argv[1])
	timeline_length_end = int(sys.argv[2])
	indexes = []
	buckets = {}
	new_papers = []
	remaining_papers = []
	for paper in papers:
		paper.timeline_of_citations=SortedDict(paper.timeline_of_citations)
		years = list(paper.timeline_of_citations.keys())
		first = 0
		last = 0
		if len(years) > 0:
			first = years[0]
			last =  years[-1]
		for i in range(first,last):
			if i not in paper.timeline_of_citations:
				paper.timeline_of_citations[i] = 0
	for time in range(timeline_length_start,timeline_length_end):
		new_papers = []
		remaining_papers = []
		for paper in papers:
			length = len(paper.timeline_of_citations)
			# t
			count0 = 0
			for r in list(paper.timeline_of_citations.values()):
				if r == 0:
					count0 += 1
			if (length == time) and (count0 < time/2):
				new_papers.append(paper)
			else:
				remaining_papers.append(paper)
		text = ""
		for paper in new_papers:
			timeline_years = list(paper.timeline_of_citations.keys())
			timeline_refs = list(paper.timeline_of_citations.values())
			line_year = ""
			line_ref  = ""
			for y in timeline_years:
				line_year += str(y)+","
			for r in timeline_refs:
				line_ref += str(r)+","
			text += line_year+"\n"
			text += line_ref+"\n"
		with open("timelines/timeline"+str(time)+"_years_refs.txt", "w") as handle:
			handle.write(text)
		end = timer()
		elapsed=end-start
		print("Finished writing timeline"+str(time)+"_years_refs in "+str(elapsed) +" seconds")
		papers = []
		papers = remaining_papers


	buckets = SortedDict(buckets)
	for bucket in buckets:
		print(bucket,buckets[bucket])



if __name__ == "__main__":
	main()
