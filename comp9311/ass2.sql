-- COMP9311 Assignment 2
-- Written by NIMA MIRSEPASSI z5437291

-- Q1_a: get details of the current Heads of Schools

create or replace view Q1_a(name, school, starting)
as
 select p.name, org.longname as school, aff.starting
   from people p
     join affiliation aff on (p.id = aff.staff)
     join orgunits org on (aff.orgunit = org.id)
  where aff.ending is null 
  and aff.isprimary = true 
  and aff.role = 1054 
  and org.utype = 2
  order by starting;

-- Q1_b: longest-serving and most-recent current Heads of Schools

create or replace view Q1_b1(status, name, school, startng)
as
select cast('Longest serving' as text), q1_a.name, q1_a.school, q1_a.starting
from q1_a
where q1_a.starting = (select min(q1_a.starting)
                                from q1_a);

create or replace view Q1_b2(status, name, school, starting)
as
select cast('Most recent' as text), q1_a.name, q1_a.school, q1_a.starting
from q1_a
where q1_a.starting = (select max(q1_a.starting)
                                from q1_a);

Create or replace view Q1_b(status, name, school, starting)
as
select * from q1_b1 
union all select * from q1_b2;

-- Q2: the subjects that used the Central Lecture Block the most

create or replace view Q2_1(subjectcode,use_rate)
as
select su.code, count(su.code)
from subjects su join courses co on (su.id = co.subject)
                 join terms t on (t.id = co.term)
                 join classes cl on (cl.course = co.id)
                 join rooms ro on (ro.id = cl.room)
                 join buildings bu on (bu.id = ro.building)
                 where bu.id = '106' and (t.year >=2007 and t.year <=2009)
group by su.code;

create or replace view Q2(subjectcode,use_rate)
as
select Q2_1.subjectcode, Q2_1.use_rate
from Q2_1
where Q2_1.use_rate = (select max(Q2_1.use_rate)
                        from Q2_1);


-- Q3: all the students who has scored HD no less than 30 time

create or replace view Q3(unsw_id, student_name)
as
select p.unswid as unsw_id, p.name as student_name
    from people p
    join courseenrolments e on (p.id = e.student)
  	where e.grade::bpchar = 'HD'::bpchar
  	group by p.unswid, p.name
 	having count(e.grade) > 30
order by unsw_id;

-- Q4: max fail rate

create or replace view q4_1(course, mark)
as
select course, mark
from courseenrolments
where mark is not null 
order by course;

create or replace view  q4_2(course, mark)
as
select course, count(mark)
from q4_1
group by course
having count(mark)> '50';

create or replace view q4_3(course, mark)
as
select distinct ce.course, ce.mark
from courseenrolments ce, courses co, classes cl
where ce.course = co.id
  and co.id = cl.course
  and cl.startdate > '2007-01-01'
  and cl.enddate < '2007-12-31'
  and ce.mark is not null
order by ce.course;

create or replace view q4_4(course, mark)
as
select course, count(student)
from courseenrolments
where mark is not null
    and mark < 50
group by course;

create or replace view q4_5(course, mark)
as
select d.course, d.mark
from q4_2 d
where d.course in (select cl.course from classes cl
        where (cl.course = d.course)
        and cl.startdate >= '2007-01-01'
        and cl.enddate <= '2007-12-31');

create or replace view q4_6(course, mark)
as
select c.course, c.mark
from q4_4 c
where c.course in (select cl.course from classes cl
        where cl.course = c.course
        and cl.startdate >= '2007-01-01'
        and cl.enddate <= '2007-12-31');

create or replace view q4_7(course_id, mark)
as
select c.course, c.mark::float / p.mark::float as failureratio
from q4_6 c, q4_5 p
where c.course = p.course;

create or replace view q4(course_id)
as
select course_id
from q4_7
where mark = (select max(mark)
                from q4_7);

--- Q5: total FTE students per term from 2001 S1 to 2010 S2 

create or replace view q5_1(id, courseid, coursesubject, uoc, studentno )
as
select t.id, c.id, c.subject, cast(s.uoc as numeric(6,1)), ce.student
from terms t join courses c on (t.id = c.term)
             join subjects s on (s.id = c.subject)
             join courseenrolments ce on (ce.course = c.id)
where (t.year  between 2000 and 2010)
and (t.sess = 'S1' or t.sess = 'S2');

create or replace view q5(term, nstudes, fte)
as
select termname(id), count(distinct studentno), cast(sum(q5_1.uoc)/24 as numeric(6,1))
from q5_1
group by q5_1.id;

-- Q6: subjects with > 30 course offerings and no staff recorded

create or replace view q6(subject, nofferings)
as
select (su.code||' '||su.name) as scn , count(su.code)
from courses co join subjects su on (co.subject = su.id)
                left outer join coursestaff cs on (cs.course = co.id)
group by scn
having (count(su.code)>30 and count(cs.staff)=0);

-- Q7:  which rooms have a given facility

--create or replace function
--	Q7(text) returns setof FacilityRecord
--as $$
--... one SQL statement, possibly using other views defined by you ...
--$$ language sql
--;

-- Q8: semester containing a particular day

create or replace function Q8(_day date) returns text 
as $$
declare
termidno text;
begin
		select t.id, t.year, t.sess, t.starting, t.ending
        from terms t
        where _day >= t.starting
        and _day <= t.ending;
        return termidno;
end;
$$ language plpgsql
;

-- Q9: transcript with variations

--create or replace function
--	q9(_sid integer) returns setof TranscriptRecord
--as $$
--declare
--	... PLpgSQL variable delcarations ...
--begin
--	... PLpgSQL code ...
--end;
--$$ language plpgsql
--;
