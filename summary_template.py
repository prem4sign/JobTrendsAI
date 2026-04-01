def generate_market_summary(role, location, jobs_df, top_market, tools_df, top_industries, top_companies, jobs_report) -> str:
    total_jobs = len(jobs_df)
    
    # Get top skills
    top_skill_names = top_market["Skill"].head(5).tolist() if not top_market.empty else []
    top_skills_text = ", ".join(top_skill_names) if top_skill_names else "various technical skills"
    
    # Get top tools
    top_tool_names = tools_df["Skill"].head(5).tolist() if not tools_df.empty else []
    top_tools_text = ", ".join(top_tool_names) if top_tool_names else "industry-standard tools"
    
    # Get top industry
    top_industry = top_industries["Industry"].iloc[0] if not top_industries.empty else "Technology"
    top_industry_count = top_industries["size"].iloc[0] if not top_industries.empty else 0
    industry_percentage = round((top_industry_count / total_jobs * 100)) if total_jobs > 0 else 0
    
    # Get experience insights
    exp_data = jobs_report["Required Experience"].value_counts()
    most_common_exp = exp_data.index[0] if len(exp_data) > 0 else "Not mentioned"
    
    # Get level distribution
    level_data = jobs_report["Level"].value_counts()
    senior_count = level_data.get("Senior", 0)
    junior_count = level_data.get("Junior", 0)
    mid_count = level_data.get("Mid-level", 0)
    
    # Get top company
    top_company = top_companies["company"].iloc[0] if not top_companies.empty else "various"
    
    summary = f"""**Market Overview for {role.strip()} in {location.strip()}**

Our analysis found **{total_jobs} active job openings** matching your search criteria. Here's what the market is telling us:

**Domain Expertise in Demand:** Employers are primarily seeking professionals with expertise in {top_skills_text}. These core competencies appear across the majority of job descriptions and are critical for competitive positioning in this market.

**Technology Stack:** The most sought-after technical tools and platforms include {top_tools_text}. Organizations are investing heavily in these technologies, making proficiency in these areas a significant advantage.

**Industry Focus:** The **{top_industry}** sector dominates the market with {top_industry_count} openings ({industry_percentage}% of total), suggesting strong growth and hiring momentum in this vertical. This indicates where the most opportunities lie for career advancement.

**Experience Requirements:** Most positions require {most_common_exp}, suggesting this is a stable market with opportunities for both experienced professionals and those looking to transition.

**Seniority Distribution:** The market shows a diverse mix with {senior_count} Senior roles, {mid_count} Mid-level positions, and {junior_count} Junior opportunities. This indicates varied career path options across different experience levels.

**Top Hiring Companies:** Leading the hiring charge are organizations like {top_company} and other major players, suggesting consolidation around established market leaders offering more stable employment prospects.

This data reveals a vibrant market with clear skill requirements and multiple entry points regardless of your current experience level. The demand across various seniority levels indicates strong organizational investment and growth opportunities."""
    
    return summary
